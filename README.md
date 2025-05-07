# Integrated Crypto Trading System (Paper 1 + Paper 2)

This project integrates two complementary trading systems:
1. **Paper 1**: Multimodal trading system using candlestick patterns and news sentiment
2. **Paper 2**: Adaptive ensemble controller with dynamic strategy weighting

## Directory Structure

```
paper3/
├── paper1/                          # Paper 1 code (multimodal trading)
│   ├── run_paper1_multimodal_test.py # 멀티모달 트레이딩 시뮬레이션 실행기
│   └── __init__.py                   # 패키지 초기화 파일
├── paper2/                          # Paper 2 code (ensemble controller)
│   ├── run_paper2_ensemble.py        # 앙상블 컨트롤러 시뮬레이션 실행기
│   └── __init__.py                   # 패키지 초기화 파일
├── integrated/                      # Integrated components
│   ├── comparative_analyzer.py       # 결과 비교 및 시각화
│   └── __init__.py                   # 패키지 초기화 파일
├── main.py                          # Main entry point to run both systems
├── monitor_simulation.py            # 시뮬레이션 모니터링 도구
├── requirements.txt                 # Project dependencies
├── run.bat                          # Windows 실행 스크립트
├── run.sh                           # Linux/Mac 실행 스크립트
└── results/                         # Results directory
    └── integrated_run_YYYYMMDD_HHMMSS/
        ├── paper1_results/          # Paper 1 simulation results
        ├── paper2_results/          # Paper 2 simulation results
        └── comparative_analysis/    # Comparative analysis results
            └── visualizations/      # Comparative charts and visualizations
```

## Features

- **One-Click Execution**: Run both trading systems with a single command
- **Comparative Analysis**: Automatically compare performance metrics between systems
- **Visualization**: Generate rich visualizations showing relative performance
- **Failsafe Mechanism**: Sample data generation if components fail
- **Logging**: Comprehensive logging of all operations

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/drl-candlesticks-trader.git
cd drl-candlesticks-trader/paper3
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the integrated system with a single command:

```
python main.py
```

This will:
1. Run the Paper 1 multimodal trading system
2. Run the Paper 2 ensemble controller 
3. Compare and visualize the results

Results will be saved in the `results/integrated_run_YYYYMMDD_HHMMSS/` directory.

## Detailed Documentation

### Paper 1: Multimodal Trading System

Combines chart pattern analysis and news sentiment to make trading decisions:
- CNN-based candlestick pattern recognition
- NLP-based news sentiment analysis
- Multimodal fusion of different data sources

See [paper1/README.md](paper1/README.md) for more details.

### Paper 2: Adaptive Ensemble Controller

Uses market state detection and dynamic weighting of strategies:
- Market state classification algorithm
- Dynamic weight adjustment based on market conditions
- Ensemble integration of multiple strategies

See [paper2/README.md](paper2/README.md) for more details.

### Integrated System

The integration provides:
- Unified execution of both systems
- Performance comparison framework
- Visualizations of relative strengths/weaknesses
- Enhanced portfolio management options

## License

MIT

## Acknowledgments

- Research team members
- Open source community
- Data providers 