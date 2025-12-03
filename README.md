# Insurance Risk Analytics & Predictive Modeling

## Project Overview

This project is part of the **AlphaCare Insurance Solutions (ACIS)** data analytics initiative to analyze historical insurance claim data and optimize marketing strategies by identifying low-risk segments for premium reduction opportunities.

### Business Objective

Develop cutting-edge risk and predictive analytics for car insurance planning and marketing in South Africa. The analysis aims to:
- Discover "low-risk" target segments for premium optimization
- Build predictive models for optimal premium pricing
- Support data-driven marketing strategy decisions

### Project Structure

```
week3/
├── data/                    # Data files (gitignored)
│   └── MachineLearningRating_v3.txt
├── src/                     # Source code
│   ├── __init__.py
│   ├── data_loader.py      # Data loading utilities
│   ├── eda.py              # Exploratory Data Analysis
│   └── utils.py            # Helper functions
├── notebooks/              # Jupyter notebooks for analysis
│   └── task1_eda.ipynb
├── outputs/                # Generated outputs (plots, reports)
│   ├── figures/
│   └── reports/
├── tests/                  # Unit tests
├── .github/
│   └── workflows/
│       └── ci.yml         # CI/CD pipeline
├── requirements.txt        # Python dependencies
└── README.md
```

## Data Description

**Historical Period**: February 2014 to August 2015

**Data Source**: `data/MachineLearningRating_v3.txt` (pipe-delimited format)

### Key Columns

- **Policy Information**: UnderwrittenCoverID, PolicyID, TransactionMonth
- **Client Demographics**: Gender, MaritalStatus, Language, Citizenship, LegalType
- **Location**: Province, PostalCode, MainCrestaZone, SubCrestaZone
- **Vehicle Details**: Make, Model, VehicleType, RegistrationYear, Cubiccapacity, Kilowatts
- **Plan Details**: SumInsured, CoverType, CoverCategory, TermFrequency
- **Financial**: TotalPremium, TotalClaims, CalculatedPremiumPerTerm

## Tasks

### Task 1: Project Planning - EDA & Statistics
- [x] Git repository setup
- [x] GitHub Actions CI/CD
- [ ] Exploratory Data Analysis (EDA)
- [ ] Statistical analysis and hypothesis testing preparation
- [ ] Data quality assessment

### Task 2: A/B Hypothesis Testing
- [ ] Test risk differences across provinces
- [ ] Test risk differences between zipcodes
- [ ] Test margin differences between zip codes
- [ ] Test risk differences between Women and Men

### Task 3: Machine Learning & Statistical Modeling
- [ ] Linear regression per zipcode (predict TotalClaims)
- [ ] ML model for optimal premium prediction
- [ ] Feature importance analysis
- [ ] Model evaluation and reporting

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- GitHub account

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd week3
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis

1. **EDA Analysis**:
```bash
python src/eda.py
```

2. **Jupyter Notebook**:
```bash
jupyter notebook notebooks/task1_eda.ipynb
```

## Key Metrics

- **Loss Ratio**: TotalClaims / TotalPremium
- **Risk Segments**: By Province, VehicleType, Gender, PostalCode
- **Profitability**: Premium vs Claims analysis

## References

- [50 Common Insurance Terms](https://cornerstoneinsurance.co.za/50-common-insurance-terms-and-what-they-mean/)
- A/B Testing methodologies
- Statistical modeling techniques

## Contributing

This is a training project for Kifiya AI Mastery Training Week 3.

## License

Educational project for training purposes.

