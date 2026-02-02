#!/bin/bash
# ==============================================================================
# ManaMetrics - Unified Command Script
# ==============================================================================
# This script provides a unified interface for all ManaMetrics operations:
# - Data collection from Scryfall API
# - ETL pipeline (PySpark)
# - Model training (Baseline, Deep Learning, Hybrid)
# - Testing and interpretability analysis
#
# Usage: ./manametrics.sh <command> [options]
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# Configuration
# ==============================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths
DATA_RAW_DIR="data/raw"
DATA_PROCESSED_DIR="data/processed"
MODELS_DIR="models"
DEFAULT_DATA_TYPE="oracle_cards"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}======================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ==============================================================================
# Setup and Environment
# ==============================================================================

setup_environment() {
    print_header "Setting Up Environment"
    
    # Set PYTHONPATH to current directory
    export PYTHONPATH=.
    print_success "PYTHONPATH set to current directory"
    
    # Check for Java (required for PySpark)
    if command -v java &> /dev/null; then
        print_success "Java found: $(java -version 2>&1 | head -n 1)"
    else
        print_warning "Java not found. PySpark ETL may not work."
        print_warning "Install Java 11+ for PySpark support"
    fi
    
    # Check for Python virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "Virtual environment active: $VIRTUAL_ENV"
    else
        print_warning "No virtual environment detected"
        print_warning "Consider activating venv: source .venv_312/Scripts/activate"
    fi
    
    echo ""
}

# ==============================================================================
# Data Collection
# ==============================================================================

collect_data() {
    print_header "Collecting Data from Scryfall"
    
    local data_type="${1:-$DEFAULT_DATA_TYPE}"
    echo "Data type: $data_type"
    
    python src/data/collect.py --type "$data_type" --output "$DATA_RAW_DIR"
    
    print_success "Data collection complete"
}

# ==============================================================================
# ETL Pipeline
# ==============================================================================

run_etl() {
    print_header "Running ETL Pipeline"
    
    local input_file="${1:-$DATA_RAW_DIR/$DEFAULT_DATA_TYPE.json}"
    local output_file="${2:-$DATA_PROCESSED_DIR/cards.parquet}"
    
    echo "Input: $input_file"
    echo "Output: $output_file"
    
    python src/data/etl.py --input "$input_file" --output "$output_file"
    
    print_success "ETL pipeline complete"
}

# ==============================================================================
# Model Training
# ==============================================================================

train_baseline() {
    print_header "Training Baseline Models"
    
    local data_path="${1:-$DATA_PROCESSED_DIR/cards.parquet}"
    
    python src/train.py --baseline --data "$data_path" --output "$MODELS_DIR"
    
    print_success "Baseline training complete"
}

train_deep() {
    print_header "Training Deep Learning Model"
    
    local data_path="${1:-$DATA_PROCESSED_DIR/cards.parquet}"
    local epochs="${2:-3}"
    
    python src/train.py --deep --data "$data_path" --output "$MODELS_DIR" --epochs "$epochs"
    
    print_success "Deep learning training complete"
}

train_hybrid() {
    print_header "Training Hybrid Multi-Modal Model"
    
    local data_path="${1:-$DATA_PROCESSED_DIR/cards.parquet}"
    local epochs="${2:-10}"
    
    python src/train.py --hybrid --data "$data_path" --output "$MODELS_DIR" --epochs "$epochs"
    
    print_success "Hybrid model training complete"
}

train_all() {
    print_header "Training All Models (Full Pipeline)"
    
    local data_path="${1:-$DATA_PROCESSED_DIR/cards.parquet}"
    local epochs="${2:-5}"
    
    python src/train.py --all --data "$data_path" --output "$MODELS_DIR" --epochs "$epochs"
    
    print_success "Full training pipeline complete"
}

# ==============================================================================
# Testing
# ==============================================================================

run_tests() {
    print_header "Running Test Suite"
    
    pytest tests/ -v
    
    print_success "All tests passed"
}

# ==============================================================================
# Interpretability Analysis
# ==============================================================================

analyze_interpretability() {
    print_header "Running Interpretability Analysis"
    
    echo "Analyzing model interpretability with SHAP..."
    echo ""
    echo "SHAP values should already be saved in $MODELS_DIR/"
    echo "This will generate visualizations and reports."
    
    python -m src.models.interpretability
    
    print_success "Interpretability analysis complete"
    echo "Check $MODELS_DIR/ for SHAP visualizations and reports"
}

# ==============================================================================
# Full Pipeline
# ==============================================================================

run_full_pipeline() {
    print_header "Running FULL Pipeline (End-to-End)"
    
    echo "This will execute:"
    echo "  1. Data Collection"
    echo "  2. ETL Processing"
    echo "  3. Model Training (All)"
    echo "  4. Test Suite"
    echo ""
    
    # Setup
    setup_environment
    
    # 1. Collect data
    collect_data "$DEFAULT_DATA_TYPE"
    
    # 2. ETL
    run_etl
    
    # 3. Train all models
    train_all
    
    # 4. Run tests
    run_tests
    
    print_success "Full pipeline execution complete!"
}

# ==============================================================================
# Help / Usage
# ==============================================================================

show_help() {
    cat << EOF
${BLUE}ManaMetrics - Unified Command Script${NC}

${GREEN}USAGE:${NC}
    ./manametrics.sh <command> [options]

${GREEN}COMMANDS:${NC}
    ${YELLOW}setup${NC}              Setup environment (check Python, Java, venv)
    
    ${YELLOW}collect${NC} [type]     Collect data from Scryfall API
                         Types: oracle_cards (default), unique_artwork, 
                                default_cards, all_cards
    
    ${YELLOW}etl${NC} [input] [out]  Run ETL pipeline
                         Default input: data/raw/oracle_cards.json
                         Default output: data/processed/cards.parquet
    
    ${YELLOW}train-baseline${NC}     Train baseline ML models (XGBoost, RF, Ridge)
    ${YELLOW}train-deep${NC} [ep]    Train deep learning text encoder (DistilBERT)
                         Default epochs: 3
    ${YELLOW}train-hybrid${NC} [ep]  Train hybrid multi-modal model
                         Default epochs: 10
    ${YELLOW}train-all${NC} [ep]     Train all models (baseline + deep + hybrid)
                         Default epochs: 5
    
    ${YELLOW}test${NC}               Run pytest test suite
    
    ${YELLOW}interpret${NC}          Generate interpretability analysis (SHAP)
    
    ${YELLOW}full${NC}               Run complete pipeline (collect → etl → train → test)
    
    ${YELLOW}help${NC}               Show this help message

${GREEN}EXAMPLES:${NC}
    ./manametrics.sh setup
    ./manametrics.sh collect oracle_cards
    ./manametrics.sh etl
    ./manametrics.sh train-baseline
    ./manametrics.sh train-deep 5
    ./manametrics.sh test
    ./manametrics.sh full

${GREEN}ENVIRONMENT:${NC}
    Ensure you have:
    - Python 3.11+ with dependencies installed (pip install -r requirements.txt)
    - Java 11+ for PySpark (optional for ETL)
    - Virtual environment activated (recommended)

${GREEN}MORE INFO:${NC}
    See README.md for detailed documentation
EOF
}

# ==============================================================================
# Main Entry Point  
# ==============================================================================

main() {
    # If no arguments, show help
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # Parse command
    command="$1"
    shift  # Remove first argument
    
    case "$command" in
        setup)
            setup_environment
            ;;
        collect)
            setup_environment
            collect_data "$@"
            ;;
        etl)
            setup_environment
            run_etl "$@"
            ;;
        train-baseline)
            setup_environment
            train_baseline "$@"
            ;;
        train-deep)
            setup_environment
            train_deep "$@"
            ;;
        train-hybrid)
            setup_environment
            train_hybrid "$@"
            ;;
        train-all)
            setup_environment
            train_all "$@"
            ;;
        test)
            setup_environment
            run_tests
            ;;
        interpret)
            setup_environment
            analyze_interpretability
            ;;
        full)
            run_full_pipeline
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
