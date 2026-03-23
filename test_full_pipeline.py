"""
End-to-End Test Script for SentinelAI Phishing Detection Pipeline

Tests the /predict and /justify endpoints with various test cases:
- Phishing SMS with suspicious URL
- Legitimate OTP message
- Delivery notification
- Image with fake UPI screenshot (placeholder)
- Message with text and URL

Run with: python test_full_pipeline.py
Requires: Server running on http://localhost:8000
"""

import asyncio
import httpx
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


# Configuration
BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{BASE_URL}/predict"
JUSTIFY_ENDPOINT = f"{BASE_URL}/justify"

# Thresholds for PASS/FAIL evaluation
PHISHING_THRESHOLD = 0.6   # Phishing messages should score ABOVE this
LEGITIMATE_THRESHOLD = 0.5  # Legitimate messages should score BELOW this


class TestType(Enum):
    PHISHING = "phishing"
    LEGITIMATE = "legitimate"


@dataclass
class TestCase:
    """A test case for the pipeline."""
    name: str
    description: str
    expected_type: TestType
    request_payload: Dict[str, Any]


@dataclass
class TestResult:
    """Result of a single test case."""
    test_case: TestCase
    prediction_response: Optional[Dict[str, Any]]
    justification: Optional[str]
    passed: bool
    error: Optional[str]


# Placeholder base64 for a fake UPI screenshot (1x1 pixel PNG)
# In production, replace with actual test image
PLACEHOLDER_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES: List[TestCase] = [
    # Test Case 1: Clear phishing SMS with suspicious URL
    TestCase(
        name="Phishing SMS - Suspicious URL",
        description="Classic phishing attempt with urgency keywords and shortened URL",
        expected_type=TestType.PHISHING,
        request_payload={
            "text": "URGENT: Your bank account has been suspended! Click here to verify immediately: bit.ly/verify-now-123 or your account will be permanently blocked within 24 hours.",
            "metadata": {
                "url": "https://bit.ly/verify-now-123",
                "sender": "VK-ALERTS",
                "timestamp": "2026-03-20T03:45:00"
            }
        }
    ),

    # Test Case 2: Legitimate OTP message from a bank
    TestCase(
        name="Legitimate OTP - Bank",
        description="Standard OTP message from a known bank sender",
        expected_type=TestType.LEGITIMATE,
        request_payload={
            "text": "Your OTP for transaction of Rs.500 at Amazon is 847293. Valid for 10 minutes. Do not share this OTP with anyone. - HDFC Bank",
            "metadata": {
                "sender": "HDFCBK",
                "timestamp": "2026-03-20T14:30:00"
            }
        }
    ),

    # Test Case 3: Legitimate delivery notification from Amazon
    TestCase(
        name="Legitimate Delivery - Amazon",
        description="Standard delivery notification with tracking link",
        expected_type=TestType.LEGITIMATE,
        request_payload={
            "text": "Your Amazon order #402-1234567-8901234 has been shipped! Track your package: amazon.in/track/D12345678. Expected delivery: March 22.",
            "metadata": {
                "url": "https://amazon.in/track/D12345678",
                "sender": "AMAZON",
                "timestamp": "2026-03-20T10:15:00"
            }
        }
    ),

    # Test Case 4: Image of fake UPI payment screenshot
    TestCase(
        name="Phishing Image - Fake UPI Screenshot",
        description="Image containing fake UPI payment confirmation (OCR-based detection)",
        expected_type=TestType.PHISHING,
        request_payload={
            "text": "Sir I have sent the payment, please check screenshot and confirm receipt. Send my order fast.",
            "image": PLACEHOLDER_IMAGE_BASE64,
            "metadata": {
                "sender": "+919876543210",
                "timestamp": "2026-03-20T16:20:00"
            }
        }
    ),

    # Test Case 5: Phishing message with text and suspicious URL
    TestCase(
        name="Phishing SMS - Prize Scam",
        description="Classic prize/lottery scam with IP-based URL",
        expected_type=TestType.PHISHING,
        request_payload={
            "text": "Congratulations! You have won Rs.50,00,000 in our lucky draw! Click here to claim your prize NOW: http://192.168.1.100/claim?id=winner2026. Offer expires today!",
            "metadata": {
                "url": "http://192.168.1.100/claim?id=winner2026",
                "sender": "PRIZE-WIN",
                "timestamp": "2026-03-20T02:30:00"
            }
        }
    ),
]


# =============================================================================
# TEST RUNNER
# =============================================================================

async def run_single_test(client: httpx.AsyncClient, test_case: TestCase) -> TestResult:
    """Run a single test case against the API."""
    try:
        # Step 1: Call /predict
        print(f"\n{'='*60}")
        print(f"TEST: {test_case.name}")
        print(f"{'='*60}")
        print(f"Description: {test_case.description}")
        print(f"Expected: {test_case.expected_type.value.upper()}")
        print(f"\n--- Input ---")
        print(json.dumps(test_case.request_payload, indent=2, default=str))

        predict_response = await client.post(
            PREDICT_ENDPOINT,
            json=test_case.request_payload,
            timeout=30.0
        )

        if predict_response.status_code != 200:
            error_msg = f"Predict failed: {predict_response.status_code} - {predict_response.text}"
            print(f"\n--- ERROR ---")
            print(error_msg)
            return TestResult(
                test_case=test_case,
                prediction_response=None,
                justification=None,
                passed=False,
                error=error_msg
            )

        prediction = predict_response.json()
        print(f"\n--- Prediction Response ---")
        print(json.dumps(prediction, indent=2))

        # Step 2: Call /justify with the prediction
        justification = None
        try:
            justify_response = await client.post(
                JUSTIFY_ENDPOINT,
                json=prediction,
                timeout=30.0
            )
            if justify_response.status_code == 200:
                justify_data = justify_response.json()
                justification = justify_data.get("justification", "")
                print(f"\n--- Justification ---")
                print(justification)
            else:
                print(f"\n--- Justification Error ---")
                print(f"Status {justify_response.status_code}: {justify_response.text}")
        except Exception as e:
            print(f"\n--- Justification Error ---")
            print(f"Failed to get justification: {e}")

        # Step 3: Evaluate PASS/FAIL
        final_score = prediction.get("final_score", 0.5)
        decision = prediction.get("decision", "")

        if test_case.expected_type == TestType.PHISHING:
            # Phishing messages should score ABOVE threshold
            passed = final_score > PHISHING_THRESHOLD
            expected_decision = "SPAM"
        else:
            # Legitimate messages should score BELOW threshold
            passed = final_score < LEGITIMATE_THRESHOLD
            expected_decision = "HAM"

        # Print evaluation
        print(f"\n--- Evaluation ---")
        print(f"Final Score: {final_score:.4f}")
        print(f"Decision: {decision}")
        print(f"Expected: {expected_decision} (score {'>' if test_case.expected_type == TestType.PHISHING else '<'} {PHISHING_THRESHOLD if test_case.expected_type == TestType.PHISHING else LEGITIMATE_THRESHOLD})")
        print(f"Result: {'PASS' if passed else 'FAIL'}")

        return TestResult(
            test_case=test_case,
            prediction_response=prediction,
            justification=justification,
            passed=passed,
            error=None
        )

    except httpx.ConnectError:
        error_msg = f"Connection failed - is the server running at {BASE_URL}?"
        print(f"\n--- ERROR ---")
        print(error_msg)
        return TestResult(
            test_case=test_case,
            prediction_response=None,
            justification=None,
            passed=False,
            error=error_msg
        )
    except Exception as e:
        error_msg = f"Test failed with exception: {e}"
        print(f"\n--- ERROR ---")
        print(error_msg)
        return TestResult(
            test_case=test_case,
            prediction_response=None,
            justification=None,
            passed=False,
            error=error_msg
        )


async def run_all_tests() -> List[TestResult]:
    """Run all test cases."""
    print("\n" + "="*60)
    print("SENTINELAI END-TO-END PIPELINE TEST")
    print("="*60)
    print(f"Server: {BASE_URL}")
    print(f"Phishing threshold: > {PHISHING_THRESHOLD}")
    print(f"Legitimate threshold: < {LEGITIMATE_THRESHOLD}")
    print(f"Total test cases: {len(TEST_CASES)}")

    results = []

    async with httpx.AsyncClient() as client:
        # Check server health first
        try:
            health_response = await client.get(f"{BASE_URL}/health", timeout=10.0)
            if health_response.status_code == 200:
                health = health_response.json()
                print(f"\nServer Status: {health.get('status', 'unknown')}")
                print(f"Models Loaded: {health.get('models_loaded', {})}")
            else:
                print(f"\nWarning: Health check returned {health_response.status_code}")
        except Exception as e:
            print(f"\nError: Could not connect to server - {e}")
            print("Make sure the server is running: python -m uvicorn app.main:app --reload")
            return results

        # Run each test case
        for test_case in TEST_CASES:
            result = await run_single_test(client, test_case)
            results.append(result)

    return results


def print_summary(results: List[TestResult]):
    """Print test summary with statistics."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    # Separate by type
    phishing_results = [r for r in results if r.test_case.expected_type == TestType.PHISHING]
    legitimate_results = [r for r in results if r.test_case.expected_type == TestType.LEGITIMATE]

    # Calculate scores for successful predictions
    phishing_scores = [
        r.prediction_response["final_score"]
        for r in phishing_results
        if r.prediction_response is not None
    ]
    legitimate_scores = [
        r.prediction_response["final_score"]
        for r in legitimate_results
        if r.prediction_response is not None
    ]

    # Print individual results
    print("\nIndividual Results:")
    print("-" * 40)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        score = result.prediction_response.get("final_score", "N/A") if result.prediction_response else "ERROR"
        if isinstance(score, float):
            score = f"{score:.4f}"
        print(f"  [{status}] {result.test_case.name}: {score}")

    # Print statistics
    print("\n" + "-"*40)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass Rate: {passed/total*100:.1f}%")

    print("\n" + "-"*40)
    print("Score Statistics:")

    if phishing_scores:
        avg_phishing = sum(phishing_scores) / len(phishing_scores)
        print(f"  Phishing (True Positive) - Avg Score: {avg_phishing:.4f}")
        print(f"    Expected: > {PHISHING_THRESHOLD}")
        print(f"    Scores: {[f'{s:.4f}' for s in phishing_scores]}")
    else:
        print("  Phishing: No valid results")

    if legitimate_scores:
        avg_legitimate = sum(legitimate_scores) / len(legitimate_scores)
        print(f"  Legitimate (False Positive) - Avg Score: {avg_legitimate:.4f}")
        print(f"    Expected: < {LEGITIMATE_THRESHOLD}")
        print(f"    Scores: {[f'{s:.4f}' for s in legitimate_scores]}")
    else:
        print("  Legitimate: No valid results")

    # Final verdict
    print("\n" + "="*60)
    if passed == total:
        print("OVERALL RESULT: ALL TESTS PASSED")
    else:
        print(f"OVERALL RESULT: {failed} TEST(S) FAILED")
    print("="*60)


async def main():
    """Main entry point."""
    results = await run_all_tests()
    if results:
        print_summary(results)
    else:
        print("\nNo test results - server may not be running.")
        print("Start the server with: python -m uvicorn app.main:app --reload")


if __name__ == "__main__":
    asyncio.run(main())
