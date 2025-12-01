# MEO_LEO

## Overview
This project implements a **satellite routing system supporting loop detection and intelligent edge node selection**. Specifically optimized for MEO-LEO constellation routing scenarios, the system integrates a variety of advanced routing strategies and performance monitoring functionalities.

## Project Structure
```
src/
├── routing.py              # Core routing algorithms
├── satellites.py           # Satellite data structures
├── environment.py          # Environment management
├── rl_agent.py            # Reinforcement learning agent
├── main.py                # Main program entry
└── ...

examples/
└── advanced_routing_demo.py # Function demonstration

docs/
├── ADVANCED_ROUTING.md     # Detailed technical documentation
└── ADVANCED_ROUTING_SUMMARY.md # Technical summary
```

## Core Functions
### Main Routing Function
```python
def route_request_with_intelligent_edge_selection(
    src_leo_id: int,
    dst_leo_id: int,
    leos: Dict[int, LEOSatellite],
    meos: Dict[int, MEOSatellite],
    agent: RLAgent,
    max_hops: int = 25,
    max_retries: int = 3,
    use_redundant_paths: bool = True,
    load_weight: float = 0.25,
    distance_weight: float = 0.35,
    connectivity_weight: float = 0.25,
    reliability_weight: float = 0.15,
    load_threshold: float = 0.8
) -> Tuple[List[int], Dict[str, any]]
```

## How to Run
### Basic Demonstration
```bash
python -m src.main
```
