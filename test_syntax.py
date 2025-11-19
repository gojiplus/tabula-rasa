"""
Syntax and import test - verifies code structure without heavy dependencies
"""

import sys

def test_syntax():
    """Test that the production script has valid syntax"""
    print("="*80)
    print("SYNTAX AND STRUCTURE TEST")
    print("="*80)

    print("\n[1/2] Testing Python syntax...")
    try:
        with open('production_table_llm.py', 'r') as f:
            code = f.read()
            compile(code, 'production_table_llm.py', 'exec')
        print("  ✓ production_table_llm.py has valid Python syntax")
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False

    print("\n[2/2] Checking code structure...")

    # Check for required classes
    required_classes = [
        'AdvancedStatSketch',
        'AdvancedQueryExecutor',
        'Query',
        'StatisticalEncoder',
        'ProductionTableQA',
        'TableQADataset',
        'ProductionTrainer'
    ]

    with open('production_table_llm.py', 'r') as f:
        code = f.read()

    missing_classes = []
    for cls in required_classes:
        if f'class {cls}' in code:
            print(f"  ✓ Found class: {cls}")
        else:
            print(f"  ✗ Missing class: {cls}")
            missing_classes.append(cls)

    if missing_classes:
        print(f"\n  Missing classes: {missing_classes}")
        return False

    print("\n" + "="*80)
    print("SYNTAX TEST PASSED!")
    print("="*80)
    print("\nCode structure is valid. All required classes are present.")
    print("\nTo run full tests with training:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run test: python test_basic.py")
    print("  3. Run full demo: jupyter notebook demo_multiple_datasets.ipynb")
    print("="*80)

    return True

if __name__ == "__main__":
    success = test_syntax()
    sys.exit(0 if success else 1)
