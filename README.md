# SwiftRL

High-Performance Reinforcement Learning Environment with Python and Go

## Overview

SwiftRL is a Reinforcement Learning project that implements a Q-Learning agent in Python while using a Go-based environment engine for performance-oriented simulation.

The project explores the practical integration of Python and Go in a machine learning workflow. Python is responsible for the Reinforcement Learning logic, training, evaluation, and visualization, while Go provides a compiled environment engine exposed to Python through a C-compatible shared library.

The main objective is not only to implement Q-Learning, but also to investigate the performance trade-offs introduced by cross-language communication and different execution strategies.

## Key Features

- Custom GridWorld environment
- Q-Learning implementation from scratch
- Epsilon-greedy exploration
- Q-table based learning
- Python-based training pipeline
- Go-based environment engine
- Python-to-Go integration through a shared DLL
- Batch environment execution
- Model persistence
- Evaluation pipeline
- Learned policy visualization
- Python vs Go performance benchmarking
- Automated Python tests

## Architecture

```text
                    SwiftRL
                       |
              +--------+--------+
              |                 |
           Python              Go
              |                 |
        Q-Learning Agent    GridWorld Engine
              |                 |
        Training / Eval      Environment
              |                 |
              +-------+---------+
                      |
                    ctypes
                      |
                 Go Shared DLL

The Reinforcement Learning algorithm remains in Python, while the environment simulation is implemented in Go.

This separation allows the project to investigate whether moving computational workloads to a compiled language provides measurable performance benefits.

Environment

The project uses a custom 6x6 GridWorld environment.

S . . . . .
. # . # . .
. . . # . .
. # . . . .
. # . . # .
. . . . . G

Where:

S = Start
G = Goal
# = Obstacle
. = Free Cell

The agent can perform four actions:

0 = Up
1 = Down
2 = Left
3 = Right
Reward System
Event	Reward
Normal step	-1
Invalid movement / obstacle	-10
Goal reached	+100

The negative step reward encourages the agent to find shorter paths.

Reinforcement Learning

SwiftRL implements Q-Learning from scratch.

The agent maintains a Q-table:

Q(state, action)

For the 6x6 environment:

36 states × 4 actions

The Q-value update follows the standard Q-Learning formulation:

Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]

where:

α is the learning rate
γ is the discount factor
r is the received reward
s' is the next state

The agent uses an epsilon-greedy strategy to balance exploration and exploitation.

Python and Go Integration

The Go environment is compiled as a C-compatible shared library:

gridworld.dll

Python loads the library using ctypes.

The basic execution flow is:

Python Q-Learning
       |
       | action
       v
Python Environment Wrapper
       |
       | ctypes
       v
Go GridWorld Engine
       |
       | state + reward + done
       v
Python Q-Learning

This design keeps the learning algorithm in Python while allowing the environment implementation to be optimized independently.

Batch Execution

A per-step Python-to-Go call introduces Foreign Function Interface overhead.

The project therefore also implements batch execution:

Python
   |
   | actions[]
   v
Go
   |
   +-- step
   +-- step
   +-- step
   +-- ...
   |
   v
states + rewards + done
   |
   v
Python

Batch execution significantly reduces the number of cross-language calls.

Performance Benchmark

The benchmark compares three environment implementations:

Native Python environment
Go environment with per-step FFI calls
Go environment with batch execution
Results
Implementation	Steps/sec
Python Environment	2,046,711.69
Go Per-Step FFI	184,227.22
Go Batch FFI	1,479,672.34
Relative Performance
Python Environment       1.00x
Go Per-Step FFI          0.09x
Go Batch FFI             0.72x

The batch implementation achieved approximately:

1,479,672 / 184,227 ≈ 8.03x

the throughput of the per-step Go FFI implementation.

Performance Analysis

The benchmark demonstrates an important engineering result.

Moving a lightweight operation from Python to Go does not automatically make it faster.

The initial design performed a Python-to-Go boundary crossing for every environment step:

Python → Go → Python

Because the GridWorld computation itself is extremely small, the FFI overhead became the dominant cost.

Batch execution reduced this overhead substantially by allowing Go to process many environment steps during a single Python-to-Go call.

However, the batch implementation still did not exceed the native Python implementation in this particular workload.

This result is expected for a small environment and demonstrates why performance optimization should be driven by profiling and benchmarking rather than assumptions.

Project Structure
SwiftRL/
│
├── README.md
├── requirements.txt
├── .gitignore
├── go.mod
│
├── python/
│   ├── __init__.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   └── q_learning.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── benchmark.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualize.py
│   │
│   └── environment.py
│
├── go/
│   ├── main.go
│   │
│   ├── environment/
│   │   └── gridworld.go
│   │
│   └── bindings/
│       └── main.go
│
├── configs/
│   └── config.yaml
│
├── notebooks/
│   └── experiments.ipynb
│
├── results/
│   ├── models/
│   ├── plots/
│   └── benchmarks/
│
└── tests/
    ├── python/
    │   ├── test_q_learning.py
    │   └── test_environment.py
    │
    └── go/
Installation

Clone the repository and enter the project directory.

Install the Python dependencies:

pip install -r requirements.txt

Verify Go:

go version
Build the Go Environment

The Go environment is compiled as a shared library.

On Windows:

go build -buildmode=c-shared -o go\bindings\gridworld.dll .\go\bindings

This generates:

go/bindings/gridworld.dll
go/bindings/gridworld.h
Training

Run the Q-Learning training pipeline:

python -m python.training.train

The training process:

Resets the Go environment.
Selects an action using the Q-Learning agent.
Sends the action to the Go environment.
Receives the next state and reward.
Updates the Q-table.
Repeats until the episode terminates.
Decays epsilon after each episode.

The trained Q-table is saved under:

results/models/q_learning.npy
Evaluation

Evaluate the trained agent:

python -m python.evaluation.evaluate

The evaluation reports:

Success rate
Average reward
Average steps
Best path
Best number of steps

Exploration is disabled during evaluation so that the learned policy can be evaluated deterministically.

Visualization

Visualize the learned policy:

python -m python.visualization.visualize

The generated visualization is saved under:

results/plots/learned_policy.png
Benchmarking

Run the environment benchmark:

python -m python.training.benchmark

The benchmark measures environment throughput in steps per second.

It compares:

Python Environment
Go Environment
Go Batch Environment
Testing

Run all Python tests:

python -m pytest tests/python

The test suite covers:

Q-Learning initialization
Action selection
Q-value updates
Epsilon decay
Environment reset
Environment transitions
Batch execution
Technologies
Python
Python 3
NumPy
PyTorch
Matplotlib
Pandas
PyYAML
pytest
Go
Go
Standard library
C-compatible shared library
ctypes integration
Design Principles

SwiftRL follows several engineering principles:

Implement core Reinforcement Learning algorithms from scratch.
Keep the RL logic independent from the environment implementation.
Separate experimentation from performance-critical components.
Benchmark before optimizing.
Avoid assuming that a compiled language is automatically faster.
Minimize cross-language communication overhead.
Keep experiments reproducible where possible.
Use automated tests for core functionality.
Limitations

The current implementation is intentionally focused on a small GridWorld environment.

The Go environment does not automatically provide a performance advantage for every workload. In this project, the per-step FFI implementation was significantly slower because the cost of Python-to-Go communication dominated the lightweight environment computation.

Batch execution reduced this overhead substantially, but the native Python implementation remained faster for this particular workload.

The benchmark should therefore be interpreted as an investigation into cross-language integration rather than a universal comparison between Python and Go.

Future Improvements

Potential future directions include:

Parallel environment simulation using Go goroutines
Larger and more computationally expensive environments
Vectorized environment execution
Improved Python-Go communication
Shared memory based communication
Parallel Q-Learning experiments
Deep Q-Network implementation
Experience replay
Target networks
Continuous environments
Multi-agent reinforcement learning
More comprehensive performance profiling
Learning Outcomes

This project demonstrates practical understanding of:

Reinforcement Learning fundamentals
Q-Learning
Epsilon-greedy exploration
Q-tables
Environment design
Model persistence
Evaluation methodology
Python-Go interoperability
C-compatible Go libraries
Foreign Function Interface overhead
Batch processing
Performance benchmarking
Profiling-driven optimization
Software architecture for ML systems
Conclusion

SwiftRL demonstrates a hybrid approach to Reinforcement Learning in which Python provides the learning and experimentation layer while Go provides a compiled environment engine.

The most important result of the project is not simply that Go can execute compiled code faster. Instead, the project demonstrates how system boundaries, communication overhead, workload size, and execution strategy affect real-world performance.

The benchmark showed that per-step Python-to-Go communication introduced substantial overhead, while batch execution significantly reduced that cost.

This provides a practical foundation for future Reinforcement Learning systems where Go can be applied to larger simulations, parallel environment execution, and computationally intensive workloads.

License

This project is intended for educational and research purposes.