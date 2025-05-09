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

## Data Download

프로젝트에 필요한 데이터셋은 다음 Google Drive 링크에서 다운로드할 수 있습니다:
- **Google Drive**: [https://drive.google.com/drive/folders/1vHxKgrkjguXfgmIOUWqbbSdD1XXDnbSK?usp=sharing](https://drive.google.com/drive/folders/1vHxKgrkjguXfgmIOUWqbbSdD1XXDnbSK?usp=sharing)
- 캔들 차트 이미지(224X224) 데이터 용량: 8.19GB/369,456장 | 2021-10-12 ~ 2023-12-19
- 암호화폐 뉴스 기사(감성분석) 데이터 용량: 12.6MB/31,038개 |  2021-10-12 ~ 2023-12-19
다운로드한 데이터 파일을 다음과 같이 배치하세요:
- `paper1/data/` - Paper 1 시스템용 데이터
- `paper2/data/` - Paper 2 시스템용 데이터
- `data/` - 통합 시스템용 추가 데이터

데이터셋에는 다음이 포함됩니다:
- 암호화폐 가격 데이터 (다양한 기간 및 타임프레임)
- 뉴스 및 감성 분석 데이터
- 테스트 및 검증용 데이터셋
- 시장 상태 레이블이 있는 참조 데이터

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
