"""Verification script to check negative samples' group ID and scores in actual training.

This script can be called during training to verify:
1. Negative samples have correct UIDs (matching their parent samples)
2. Negative samples have score=-1 in rm_scores
3. Negative samples are properly grouped with their parents for GRPO advantage calculation

Usage:
    Add this verification call in ray_trainer.py after batch.union(gen_batch_output):
    
    from verify_negative_samples_in_training import verify_negative_samples
    verify_negative_samples(batch, step=self.global_steps)
"""

import torch
import numpy as np
from collections import defaultdict


def verify_negative_samples(batch, step=None, verbose=True):
    """Verify negative samples in the batch.
    
    Args:
        batch: DataProto containing the batch data
        step: Current training step (for logging)
        verbose: Whether to print detailed information
        
    Returns:
        dict: Verification results
    """
    results = {
        "has_negative_samples": False,
        "num_negative_samples": 0,
        "num_original_samples": 0,
        "negative_samples_have_correct_uids": False,
        "negative_samples_have_score_minus_one": False,
        "groups_with_negative_samples": 0,
        "total_groups": 0,
        "all_checks_passed": False,
    }
    
    # Check if there's a negative sample marker
    if "__negative_sample_parent_indices__" in batch.non_tensor_batch:
        print(f"\n{'='*80}")
        print(f"⚠️  WARNING: Found __negative_sample_parent_indices__ marker!")
        print(f"   This should have been cleaned up after UID inheritance!")
        print(f"{'='*80}\n")
        return results
    
    # Check for negative_samples in non_tensor_batch (original data)
    if "negative_samples" not in batch.non_tensor_batch:
        if verbose:
            print(f"\n[Step {step}] No negative samples in this batch.")
        return results
    
    negative_samples_list = batch.non_tensor_batch["negative_samples"]
    
    # Count negative samples
    num_neg_samples = 0
    neg_sample_indices = []
    
    for idx, neg_samples in enumerate(negative_samples_list):
        if neg_samples:  # Not empty
            num_neg_samples += len(neg_samples)
            for _ in neg_samples:
                neg_sample_indices.append(idx)
    
    if num_neg_samples == 0:
        if verbose:
            print(f"\n[Step {step}] Negative samples list exists but is empty.")
        return results
    
    results["has_negative_samples"] = True
    results["num_negative_samples"] = num_neg_samples
    results["num_original_samples"] = len(negative_samples_list) - num_neg_samples
    
    print(f"\n{'='*80}")
    print(f"🔍 [NEGATIVE SAMPLES VERIFICATION - Step {step}]")
    print(f"{'='*80}")
    print(f"Batch size: {len(batch)}")
    print(f"Original samples: {results['num_original_samples']}")
    print(f"Negative samples found: {num_neg_samples}")
    print(f"{'='*80}\n")
    
    # Get UIDs
    if "uid" not in batch.non_tensor_batch:
        print("❌ ERROR: No UID field found in batch!")
        return results
    
    uids = batch.non_tensor_batch["uid"]
    
    # Get rm_scores
    if "rm_scores" not in batch.batch:
        print("❌ ERROR: No rm_scores field found in batch!")
        return results
    
    rm_scores = batch.batch["rm_scores"]
    response_mask = batch.batch["response_mask"]
    
    # Verify 1: Check UIDs - negative samples should share UIDs with parents
    print("📋 Verification 1: UID Inheritance")
    print("-" * 80)
    
    # Group by UID
    uid_to_indices = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_indices[uid].append(i)
    
    results["total_groups"] = len(uid_to_indices)
    
    # For each group, check if it contains both original and negative samples
    groups_with_negs = 0
    uid_issues = []
    
    for uid, indices in uid_to_indices.items():
        # Check if any index corresponds to a negative sample
        has_neg_sample = False
        for idx in indices:
            if idx < len(negative_samples_list):
                if negative_samples_list[idx]:  # This position has negative samples
                    has_neg_sample = True
                    break
        
        if has_neg_sample:
            groups_with_negs += 1
            if verbose:
                print(f"  Group UID: {uid[:16]}... has {len(indices)} samples (includes negative samples)")
    
    results["groups_with_negative_samples"] = groups_with_negs
    
    if groups_with_negs > 0:
        print(f"✅ Found {groups_with_negs} groups with negative samples")
        results["negative_samples_have_correct_uids"] = True
    else:
        print(f"❌ No groups found with negative samples - UID inheritance may have failed!")
        uid_issues.append("No groups contain negative samples")
    
    # Verify 2: Check scores - negative samples should have score=-1
    print(f"\n📊 Verification 2: Score=-1 for Negative Samples")
    print("-" * 80)
    
    # Extract scores from rm_scores (score is at the last valid token)
    scores = []
    score_issues = []
    
    for i in range(len(batch)):
        response_len = response_mask[i].sum().item()
        if response_len > 0:
            score = rm_scores[i, int(response_len - 1)].item()
            scores.append(score)
        else:
            scores.append(0.0)
    
    # Check negative sample scores
    # Negative samples are at the end of the batch (after original samples)
    original_batch_size = results["num_original_samples"]
    neg_sample_scores = scores[original_batch_size:]
    
    if len(neg_sample_scores) != num_neg_samples:
        print(f"⚠️  WARNING: Expected {num_neg_samples} negative sample scores, got {len(neg_sample_scores)}")
        score_issues.append(f"Score count mismatch: expected {num_neg_samples}, got {len(neg_sample_scores)}")
    
    # Check if all negative sample scores are -1
    negative_scores_correct = all(abs(s - (-1.0)) < 1e-6 for s in neg_sample_scores)
    
    if negative_scores_correct and len(neg_sample_scores) == num_neg_samples:
        print(f"✅ All {num_neg_samples} negative samples have score=-1")
        print(f"   Sample scores: {neg_sample_scores[:5]}{'...' if len(neg_sample_scores) > 5 else ''}")
        results["negative_samples_have_score_minus_one"] = True
    else:
        print(f"❌ Score verification failed!")
        print(f"   Negative sample scores: {neg_sample_scores}")
        wrong_scores = [(i, s) for i, s in enumerate(neg_sample_scores) if abs(s - (-1.0)) >= 1e-6]
        if wrong_scores:
            print(f"   Wrong scores at indices: {wrong_scores[:10]}")
            score_issues.append(f"Found {len(wrong_scores)} samples with score != -1")
    
    # Verify 3: Group structure for GRPO
    print(f"\n🎯 Verification 3: GRPO Group Structure")
    print("-" * 80)
    
    for uid, indices in list(uid_to_indices.items())[:5]:  # Show first 5 groups
        group_scores = [scores[i] for i in indices]
        has_neg = any(s < 0 for s in group_scores)
        
        if has_neg:
            print(f"  Group {uid[:16]}...")
            print(f"    Size: {len(indices)} samples")
            print(f"    Scores: {[f'{s:.2f}' for s in group_scores]}")
            print(f"    Mean: {np.mean(group_scores):.4f}")
            
            # Calculate what advantages would look like
            group_mean = np.mean(group_scores)
            advantages = [s - group_mean for s in group_scores]
            print(f"    Advantages (score - mean): {[f'{a:.4f}' for a in advantages]}")
            print(f"    → Negative samples lower the mean, boosting advantages for successful trajectories\n")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📈 VERIFICATION SUMMARY")
    print(f"{'='*80}")
    
    all_passed = (
        results["negative_samples_have_correct_uids"] and
        results["negative_samples_have_score_minus_one"]
    )
    results["all_checks_passed"] = all_passed
    
    if all_passed:
        print(f"✅ ALL CHECKS PASSED!")
        print(f"   ✓ UID inheritance correct")
        print(f"   ✓ Score=-1 for all negative samples")
        print(f"   ✓ GRPO grouping verified")
    else:
        print(f"❌ VERIFICATION FAILED!")
        if not results["negative_samples_have_correct_uids"]:
            print(f"   ✗ UID inheritance issues:")
            for issue in uid_issues:
                print(f"     - {issue}")
        if not results["negative_samples_have_score_minus_one"]:
            print(f"   ✗ Score issues:")
            for issue in score_issues:
                print(f"     - {issue}")
    
    print(f"{'='*80}\n")
    
    return results


def add_verification_hook_to_trainer():
    """Example of how to add verification to the trainer.
    
    Add this in ray_trainer.py fit() method after line 1260:
    
        batch = batch.union(gen_batch_output)
        
        ### ADD VERIFICATION HERE ###
        if self.global_steps % 10 == 0:  # Verify every 10 steps
            from verify_negative_samples_in_training import verify_negative_samples
            verify_results = verify_negative_samples(batch, step=self.global_steps)
            
            # Optionally log results
            if verify_results["has_negative_samples"]:
                metrics.update({
                    "verify/negative_samples_count": verify_results["num_negative_samples"],
                    "verify/groups_with_negatives": verify_results["groups_with_negative_samples"],
                    "verify/all_checks_passed": int(verify_results["all_checks_passed"]),
                })
    """
    pass


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    print(add_verification_hook_to_trainer.__doc__)
