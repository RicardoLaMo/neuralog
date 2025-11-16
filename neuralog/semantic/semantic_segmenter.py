"""
Semantic Segmenter - Event-based segmentation instead of chunking.

Replaces naive text chunking with intelligent segmentation based on:
- Semantic coherence (events, scenes, topics)
- Entity continuity
- Temporal boundaries
- Narrative structure

This is much more effective than fixed-size or sentence-based chunking.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from loguru import logger


@dataclass
class SemanticSegment:
    """
    Coherent semantic segment.

    Unlike arbitrary chunks, segments represent meaningful units:
    - Events or event sequences
    - Scenes or situations
    - Topical units
    - Dialogue exchanges
    """
    segment_id: str
    segment_type: str  # "event", "scene", "topic", "dialogue"
    text: str
    start_pos: int  # Character position in source
    end_pos: int
    entities: List[str] = None
    temporal_markers: List[str] = None
    coherence_score: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.temporal_markers is None:
            self.temporal_markers = []
        if self.metadata is None:
            self.metadata = {}


class SemanticSegmenter:
    """
    Segments text based on semantic coherence instead of arbitrary chunking.

    Strategies:
    1. Event-based: Segment by events and actions
    2. Entity-based: Segment by entity continuity
    3. Temporal-based: Segment by time markers
    4. Topic-based: Segment by semantic topics
    5. Hybrid: Combine multiple signals

    This provides much better context for LLMs than naive chunking.
    """

    def __init__(
        self,
        llm_interface=None,
        strategy: str = "hybrid",
        max_segment_length: int = 1000
    ):
        """
        Initialize Semantic Segmenter.

        Args:
            llm_interface: Optional LLM for semantic understanding
            strategy: Segmentation strategy
            max_segment_length: Maximum characters per segment (soft limit)
        """
        self.llm_interface = llm_interface
        self.strategy = strategy
        self.max_segment_length = max_segment_length

        logger.info(f"SemanticSegmenter initialized with strategy: {strategy}")

    def segment(
        self,
        text: str,
        strategy: Optional[str] = None
    ) -> List[SemanticSegment]:
        """
        Segment text semantically.

        Args:
            text: Input text
            strategy: Override default strategy

        Returns:
            List of semantic segments
        """
        strategy = strategy or self.strategy

        logger.info(f"Segmenting text (length: {len(text)}) with strategy: {strategy}")

        if strategy == "event":
            segments = self._segment_by_events(text)
        elif strategy == "entity":
            segments = self._segment_by_entities(text)
        elif strategy == "temporal":
            segments = self._segment_by_temporal(text)
        elif strategy == "topic":
            segments = self._segment_by_topic(text)
        elif strategy == "hybrid":
            segments = self._segment_hybrid(text)
        else:
            # Fallback to paragraph-based (better than fixed chunks)
            segments = self._segment_by_paragraphs(text)

        logger.info(f"Created {len(segments)} semantic segments")
        return segments

    def _segment_by_events(self, text: str) -> List[SemanticSegment]:
        """
        Segment by events and actions.

        Identifies event boundaries using:
        - Action verbs
        - Temporal markers ("then", "after", "next")
        - Narrative transitions
        """
        if not self.llm_interface:
            logger.warning("Event segmentation requires LLM, falling back")
            return self._segment_by_paragraphs(text)

        # Use LLM to identify event boundaries
        prompt = f"""Identify event boundaries in this text. Each event is a coherent action or occurrence.

Text:
{text}

Return event boundaries as JSON:
{{
  "events": [
    {{
      "start": character_position,
      "end": character_position,
      "description": "brief event description",
      "type": "action|state_change|interaction"
    }}
  ]
}}
"""

        try:
            response = self.llm_interface.generate(prompt, temperature=0.1)
            import json

            # Parse response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response

            data = json.loads(json_str)

            segments = []
            for i, event in enumerate(data.get("events", [])):
                start = event.get("start", 0)
                end = event.get("end", len(text))

                segment = SemanticSegment(
                    segment_id=f"event_{i}",
                    segment_type="event",
                    text=text[start:end],
                    start_pos=start,
                    end_pos=end,
                    metadata={"event_type": event.get("type")}
                )
                segments.append(segment)

            return segments

        except Exception as e:
            logger.error(f"Event segmentation failed: {e}")
            return self._segment_by_paragraphs(text)

    def _segment_by_entities(self, text: str) -> List[SemanticSegment]:
        """
        Segment by entity continuity.

        Create segments where the same entities are discussed.
        Segment boundaries occur when entity focus shifts.
        """
        # Simple heuristic: segment when capitalized names change significantly
        # (Real implementation would use NER)

        sentences = re.split(r'(?<=[.!?])\s+', text)
        segments = []
        current_segment_text = []
        current_entities = set()
        start_pos = 0

        for sentence in sentences:
            # Extract potential entity mentions (capitalized words)
            entities = set(re.findall(r'\b[A-Z][a-z]+\b', sentence))

            # Check entity overlap
            if current_entities and entities:
                overlap = len(current_entities & entities) / len(current_entities | entities)
                if overlap < 0.3:  # Low overlap = new segment
                    # Create segment
                    segment_text = " ".join(current_segment_text)
                    segment = SemanticSegment(
                        segment_id=f"entity_{len(segments)}",
                        segment_type="entity",
                        text=segment_text,
                        start_pos=start_pos,
                        end_pos=start_pos + len(segment_text),
                        entities=list(current_entities)
                    )
                    segments.append(segment)

                    # Start new segment
                    current_segment_text = [sentence]
                    current_entities = entities
                    start_pos += len(segment_text) + 1
                    continue

            current_segment_text.append(sentence)
            current_entities.update(entities)

        # Add final segment
        if current_segment_text:
            segment_text = " ".join(current_segment_text)
            segment = SemanticSegment(
                segment_id=f"entity_{len(segments)}",
                segment_type="entity",
                text=segment_text,
                start_pos=start_pos,
                end_pos=start_pos + len(segment_text),
                entities=list(current_entities)
            )
            segments.append(segment)

        return segments

    def _segment_by_temporal(self, text: str) -> List[SemanticSegment]:
        """
        Segment by temporal markers.

        Segment boundaries at temporal transitions:
        - "later", "then", "after", "before"
        - Date/time mentions
        - Tense changes
        """
        # Temporal markers
        temporal_markers = [
            r'\b(later|then|after|before|next|meanwhile|simultaneously|afterwards)\b',
            r'\b(yesterday|today|tomorrow|now)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\d{4}-\d{2}-\d{2}',  # Dates
            r'\b(morning|afternoon|evening|night)\b'
        ]

        # Find temporal markers
        marker_positions = []
        for pattern in temporal_markers:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                marker_positions.append((match.start(), match.group()))

        marker_positions.sort()

        # Create segments between markers
        segments = []
        current_start = 0

        for pos, marker in marker_positions:
            if pos - current_start > 100:  # Minimum segment size
                segment = SemanticSegment(
                    segment_id=f"temporal_{len(segments)}",
                    segment_type="temporal",
                    text=text[current_start:pos].strip(),
                    start_pos=current_start,
                    end_pos=pos,
                    temporal_markers=[marker]
                )
                segments.append(segment)
                current_start = pos

        # Final segment
        if current_start < len(text):
            segment = SemanticSegment(
                segment_id=f"temporal_{len(segments)}",
                segment_type="temporal",
                text=text[current_start:].strip(),
                start_pos=current_start,
                end_pos=len(text)
            )
            segments.append(segment)

        return segments if segments else self._segment_by_paragraphs(text)

    def _segment_by_topic(self, text: str) -> List[SemanticSegment]:
        """
        Segment by semantic topics.

        Uses semantic similarity to identify topic boundaries.
        """
        # Would use embeddings and similarity for topic segmentation
        # For now, fallback to paragraphs
        return self._segment_by_paragraphs(text)

    def _segment_hybrid(self, text: str) -> List[SemanticSegment]:
        """
        Hybrid segmentation combining multiple signals.

        Uses entity continuity, temporal markers, and semantic coherence.
        """
        # Start with entity-based segmentation
        segments = self._segment_by_entities(text)

        # Refine with temporal markers
        # (Split segments that are too long and have temporal markers)
        refined_segments = []

        for segment in segments:
            if len(segment.text) > self.max_segment_length:
                # Split using temporal markers
                sub_segments = self._segment_by_temporal(segment.text)
                refined_segments.extend(sub_segments)
            else:
                refined_segments.append(segment)

        return refined_segments

    def _segment_by_paragraphs(self, text: str) -> List[SemanticSegment]:
        """
        Fallback: segment by paragraphs.

        Better than fixed chunks, respects natural boundaries.
        """
        paragraphs = text.split('\n\n')
        segments = []
        current_pos = 0

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                current_pos += 2  # \n\n
                continue

            segment = SemanticSegment(
                segment_id=f"para_{i}",
                segment_type="paragraph",
                text=para,
                start_pos=current_pos,
                end_pos=current_pos + len(para)
            )
            segments.append(segment)
            current_pos += len(para) + 2

        return segments

    def merge_small_segments(
        self,
        segments: List[SemanticSegment],
        min_size: int = 100
    ) -> List[SemanticSegment]:
        """
        Merge segments that are too small.

        Args:
            segments: Input segments
            min_size: Minimum segment size

        Returns:
            Merged segments
        """
        if not segments:
            return []

        merged = []
        current = segments[0]

        for next_seg in segments[1:]:
            if len(current.text) < min_size:
                # Merge with next
                current.text += " " + next_seg.text
                current.end_pos = next_seg.end_pos
                current.entities.extend(next_seg.entities)
                current.temporal_markers.extend(next_seg.temporal_markers)
            else:
                merged.append(current)
                current = next_seg

        merged.append(current)
        return merged
