"""
Master script to run all social inhibition mechanism tests.

Quick tests (8 hours total):
- Test A: Group size effect (4 hours)
- Test B: Blind agents (2 hours)
- Test D: Reward structure (2 hours)
- Test E: Asymmetric observation (3 hours)

Test C (multi-agent fine-tuning) should be run separately as it takes 2-3 days.
"""

import sys
from pathlib import Path
from datetime import datetime

# Import test modules
sys.path.insert(0, str(Path(__file__).parent))

from test_social_inhibition import test_a_group_size_effect, test_b_blind_agents, test_d_reward_structure
from test_e_asymmetric_observation import test_e_asymmetric_observation

def main():
    """Run all quick social inhibition tests."""
    
    # Create output directory
    Path('experiments/social_inhibition').mkdir(parents=True, exist_ok=True)
    
    start_time = datetime.now()
    
    print("="*70)
    print("SOCIAL INHIBITION MECHANISM INVESTIGATION")
    print("="*70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Background:")
    print("  Multi-agent settings reduce lying by 75%")
    print("  (53.6% → 13% lying rate)")
    print()
    print("Question: WHY does this happen?")
    print()
    print("Running 4 quick tests to identify mechanism:")
    print("  Test A: Group size effect (~4 hours)")
    print("  Test B: Blind agents (~2 hours)")
    print("  Test D: Reward structure (~2 hours)")
    print("  Test E: Asymmetric observation (~3 hours)")
    print()
    print("Total estimated time: ~8-10 hours")
    print()
    print("Note: Test C (multi-agent fine-tuning) takes 2-3 days")
    print("      and should be run separately if needed.")
    print()
    print("="*70)
    
    input("\nPress Enter to begin tests...")
    
    # Run tests
    results = {}
    
    try:
        # Test A: Group Size Effect
        print("\n\n" + "#"*70)
        print("# TEST A: GROUP SIZE EFFECT")
        print("#"*70)
        results['test_a'] = test_a_group_size_effect()
        
        # Test B: Blind Agents
        print("\n\n" + "#"*70)
        print("# TEST B: BLIND AGENTS")
        print("#"*70)
        results['test_b'] = test_b_blind_agents()
        
        # Test D: Reward Structure
        print("\n\n" + "#"*70)
        print("# TEST D: REWARD STRUCTURE")
        print("#"*70)
        results['test_d'] = test_d_reward_structure()
        
        # Test E: Asymmetric Observation
        print("\n\n" + "#"*70)
        print("# TEST E: ASYMMETRIC OBSERVATION")
        print("#"*70)
        results['test_e'] = test_e_asymmetric_observation()
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        print("Partial results saved to: experiments/social_inhibition/")
        return
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print()
    print("Results saved to: experiments/social_inhibition/")
    print()
    print("Files created:")
    print("  - test_a_group_size.csv / .png")
    print("  - test_b_blind_agents.csv / .png")
    print("  - test_d_reward_structure.csv / .png")
    print("  - test_e_asymmetric.csv / .png")
    print()
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print()
    print("Based on the results, you can now:")
    print()
    print("1. Analyze which hypothesis is supported")
    print("2. Run Test C (multi-agent fine-tuning) if needed:")
    print("   python test_c_multi_agent_finetuning.py")
    print()
    print("3. Proceed to EXPERIMENT 2 (the cool novel one!)")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
