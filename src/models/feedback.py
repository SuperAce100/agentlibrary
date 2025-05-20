import asyncio
from datetime import datetime
from openai import OpenAI
from typing import List, Tuple
from models.llms import llm_call_messages_async, async_client


class InteractionMetric:
    def __init__(self, name: str, description: str, evaluation_prompt: str):
        self.name = name
        self.description = description
        self.evaluation_prompt = evaluation_prompt


async def analyze_interaction(interaction: dict, client: OpenAI) -> List[str]:
    """Extract key aspects from an interaction that should be measured."""
    prompt = """Analyze this interaction and identify key aspects of behavior that should be measured.
    Focus on concrete, measurable behaviors. Return each aspect on a new line."""

    response = await llm_call_messages_async(
        messages=[
            {
                "role": "system",
                "content": "You are analyzing chat interactions to identify measurable behaviors.",
            },
            {"role": "user", "content": f"Interaction:\n{interaction}\n\n{prompt}"},
        ]
    )

    aspects = response.strip().split("\n")
    return [aspect.strip() for aspect in aspects if aspect.strip()]


async def cluster_into_metrics(
    aspects: List[str], client: OpenAI
) -> List[InteractionMetric]:
    """Convert behavioral aspects into concrete metrics."""
    prompt = """Given these behavioral aspects, create metrics that can measure them.
    For each metric, provide:
    1. A clear name
    2. A description of what it measures
    3. An evaluation prompt to assess this metric
    
    Format each metric as: NAME|DESCRIPTION|EVALUATION_PROMPT
    Make sure to include the | character between each part."""

    response = await llm_call_messages_async(
        messages=[
            {
                "role": "system",
                "content": "You are creating evaluation metrics from behavioral aspects.",
            },
            {"role": "user", "content": f"Aspects:\n{'\n'.join(aspects)}\n\n{prompt}"},
        ]
    )

    metrics = []
    for metric_str in response.strip().split("\n"):
        # Add error handling for malformed responses
        try:
            if "|" not in metric_str:
                # Skip lines that don't contain the separator
                continue

            parts = metric_str.split("|")
            if len(parts) < 3:
                # If we don't have enough parts, create default values
                name = parts[0].strip()
                description = (
                    parts[1].strip() if len(parts) > 1 else "No description provided"
                )
                eval_prompt = "Rate this aspect on a scale from 0 to 1"
            else:
                name, description, eval_prompt = (
                    parts[0].strip(),
                    parts[1].strip(),
                    parts[2].strip(),
                )

            metrics.append(InteractionMetric(name, description, eval_prompt))
        except Exception as e:
            print(f"Error processing metric: {metric_str}, Error: {e}")
            continue

    # If we didn't get any valid metrics, create a default one
    if not metrics and aspects:
        metrics.append(
            InteractionMetric(
                "Overall Quality",
                "General assessment of response quality",
                "Rate the overall quality on a scale from 0 to 1",
            )
        )

    return metrics


async def evaluate_coverage(
    interaction: dict, metrics: List[InteractionMetric], client: OpenAI
) -> Tuple[int, int, List[str]]:
    """Evaluate how well the metrics cover the interaction's aspects."""
    prompt = """ 
    Given this interaction and these metrics, identify:
    1. How many aspects of the interaction are covered by the metrics. Be very harsh and meticulous about uncovering aspects of the interaction that may have been missed.
    2. The total number of notable aspects in the interaction
    3. List any uncovered aspects
    
    Format: COVERED_COUNT|TOTAL_COUNT|UNCOVERED_ASPECT1;UNCOVERED_ASPECT2;..."""

    metrics_str = "\n".join([f"- {m.name}: {m.description}" for m in metrics])

    response = await llm_call_messages_async(
        messages=[
            {
                "role": "system",
                "content": "You are evaluating metric coverage of interactions.",
            },
            {
                "role": "user",
                "content": f"Interaction:\n{interaction}\nMetrics:\n{metrics_str}\n\n{prompt}",
            },
        ]
    )

    covered, total, uncovered = response.strip().split("|")
    uncovered_aspects = uncovered.split(";") if uncovered else []

    return int(covered), int(total), uncovered_aspects


async def iterative_metric_creation(interaction: dict) -> List[InteractionMetric]:
    """Iteratively create metrics until good coverage is achieved."""
    client = async_client

    # Initial state
    prev_coverage_rate: float = 0
    curr_coverage_rate: float = 0
    prev_metrics: List[InteractionMetric] = []
    curr_metrics: List[InteractionMetric] = []

    # Initial aspects
    aspects = await analyze_interaction(interaction, client)

    while True:
        print(f"Current coverage rate: {curr_coverage_rate}")
        print(f"Previous coverage rate: {prev_coverage_rate}")

        prev_metrics = curr_metrics
        prev_coverage_rate = curr_coverage_rate

        # Create new metrics from uncovered aspects
        new_metrics = await cluster_into_metrics(aspects, client)
        curr_metrics = prev_metrics + new_metrics

        # Evaluate coverage
        covered, total, uncovered_aspects = await evaluate_coverage(
            interaction, curr_metrics, client
        )

        curr_coverage_rate = covered / total

        # Break if coverage isn't improving
        if curr_coverage_rate <= prev_coverage_rate:
            break

        aspects = uncovered_aspects

    # Return the best metrics we found
    return prev_metrics


def save_metrics(metrics: List[InteractionMetric], path: str) -> None:
    """Save metrics to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{path}/metrics_{timestamp}.json"

    metrics_data = [
        {
            "name": metric.name,
            "description": metric.description,
        }
        for metric in metrics
    ]

    import json

    with open(filename, "w") as f:
        json.dump(metrics_data, f, indent=2)


async def main() -> None:
    # Example trajectory
    interaction = {
        "user": "Research the impact of climate change on biodiversity in tropical rainforests and summarize the findings in a structured report with recommendations for conservation efforts.",
        "assistant": """# Impact of Climate Change on Tropical Rainforest Biodiversity

## Introduction
Tropical rainforests are biodiversity hotspots that host approximately 50% of Earth's species while covering less than 6% of its land surface. Climate change poses significant threats to these ecosystems through multiple mechanisms.

## Key Impacts

### Temperature Increases
- Rising temperatures alter flowering and fruiting patterns
- Exceed thermal tolerance of many species
- Shift optimal habitat zones upslope

### Precipitation Changes
- Increased drought frequency stresses moisture-dependent species
- Altered rainfall patterns disrupt reproductive cycles
- More intense rainfall events increase erosion and landslides

### Ecosystem Disruption
- Phenological mismatches between interdependent species
- Breakdown of pollinator relationships
- Altered predator-prey dynamics

## Most Threatened Species

### Amphibians
- **Golden Poison Frog** (*Phyllobates terribilis*): Highly sensitive to temperature and humidity changes
- **Splendid Poison Frog** (*Dendrobates speciosus*): Limited range and specific microhabitat requirements

### Primates
- **Sumatran Orangutan** (*Pongo abelii*): Habitat fragmentation combined with climate impacts
- **Golden Lion Tamarin** (*Leontopithecus rosalia*): Vulnerable to disease spread facilitated by climate change

### Plants
- **Brazil Nut Tree** (*Bertholletia excelsa*): Disrupted pollinator relationships
- **Rafflesia** species: Specific host and climate requirements

## Conservation Recommendations

1. **Establish Climate-Resilient Protected Areas**
   - Identify and protect thermal refugia
   - Create corridors connecting habitats across elevation gradients

2. **Implement Assisted Migration Programs**
   - Develop protocols for relocating at-risk species
   - Create ex-situ conservation facilities as insurance populations

3. **Restore Degraded Habitats**
   - Prioritize reforestation with climate-resilient native species
   - Implement agroforestry systems in buffer zones

4. **Reduce Non-Climate Stressors**
   - Combat illegal logging and poaching
   - Minimize fragmentation through land-use planning

5. **Engage Indigenous Communities**
   - Incorporate traditional ecological knowledge
   - Support sustainable livelihoods that incentivize conservation

## Conclusion
The impacts of climate change on tropical rainforest biodiversity are complex and severe. Immediate action through these recommended conservation strategies is essential to preserve these irreplaceable ecosystems and their biodiversity.""",
    }

    metrics = await iterative_metric_creation(interaction)
    save_metrics(metrics, ".data/metrics")


if __name__ == "__main__":
    asyncio.run(main())
