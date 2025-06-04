#!/usr/bin/env python3
"""
Complete System Test for Churn Prediction API
Tests all components of the ML system end-to-end
"""

import requests
import json
import time
import sys
import os
from typing import Dict, Any
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
MLFLOW_BASE_URL = "http://localhost:5000"

class SystemTester:
    def __init__(self):
        self.api_base = API_BASE_URL
        self.mlflow_base = MLFLOW_BASE_URL
        self.test_results = []
        self.start_time = time.time()
        
    def log_test_result(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test results with timing"""
        status = "✅ PASS" if success else "❌ FAIL"
        duration_str = f"({duration:.3f}s)" if duration > 0 else ""
        print(f"{status} {test_name} {duration_str}")
        if details:
            print(f"   {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
    
    def test_docker_containers(self):
        """Test if Docker containers are running"""
        print("\n🐳 Testing Docker Containers...")
        
        try:
            import docker
            client = docker.from_env()
            
            # Check for API container
            try:
                api_container = client.containers.get("churn_prediction_api")
                api_running = api_container.status == "running"
                self.log_test_result("API Container Status", api_running, f"Status: {api_container.status}")
            except:
                self.log_test_result("API Container Status", False, "Container not found")
            
            # Check for MLflow container (optional)
            try:
                mlflow_container = client.containers.get("mlflow_server")
                mlflow_running = mlflow_container.status == "running"
                self.log_test_result("MLflow Container Status", mlflow_running, f"Status: {mlflow_container.status}")
            except:
                self.log_test_result("MLflow Container Status", False, "Container not found (optional)")
                
        except ImportError:
            print("   Docker library not installed. Run: pip install docker")
            self.log_test_result("Docker Library", False, "docker library not available")
        except Exception as e:
            self.log_test_result("Docker Connection", False, f"Error: {str(e)}")
    
    def test_api_connectivity(self):
        """Test basic API connectivity and health"""
        print("\n🔗 Testing API Connectivity...")
        
        # Test root endpoint
        start_time = time.time()
        try:
            response = requests.get(f"{self.api_base}/", timeout=10)
            duration = time.time() - start_time
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Status: {response.status_code}, Message: {data.get('message', 'N/A')}"
            else:
                details = f"Status: {response.status_code}"
                
            self.log_test_result("Root Endpoint", success, details, duration)
        except requests.exceptions.ConnectionError:
            duration = time.time() - start_time
            self.log_test_result("Root Endpoint", False, "Connection refused - API not running?", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Root Endpoint", False, f"Error: {str(e)}", duration)
        
        # Test health endpoint
        start_time = time.time()
        try:
            response = requests.get(f"{self.api_base}/health", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                model_loaded = health_data.get("model_loaded", False)
                success = health_data.get("status") == "healthy" and model_loaded
                details = f"Healthy: {success}, Model Loaded: {model_loaded}"
            else:
                success = False
                details = f"Status: {response.status_code}"
                
            self.log_test_result("Health Check", success, details, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Health Check", False, f"Error: {str(e)}", duration)
    
    def test_model_endpoints(self):
        """Test model-related endpoints"""
        print("\n🤖 Testing Model Endpoints...")
        
        # Test model info endpoint
        start_time = time.time()
        try:
            response = requests.get(f"{self.api_base}/model-info", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                model_info = response.json()
                model_type = model_info.get("model_type", "Unknown")
                feature_count = model_info.get("feature_count", "Unknown")
                success = "XGB" in str(model_type) or "RandomForest" in str(model_type)
                details = f"Model: {model_type}, Features: {feature_count}"
            else:
                success = False
                details = f"Status: {response.status_code}"
                
            self.log_test_result("Model Info", success, details, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Model Info", False, f"Error: {str(e)}", duration)
        
        # Test feature engineering
        start_time = time.time()
        try:
            response = requests.get(f"{self.api_base}/test-features", timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                features_info = response.json()
                original = features_info.get("original_features", 0)
                processed = features_info.get("processed_features", 0)
                success = processed > original
                details = f"Original: {original}, Processed: {processed}"
            else:
                success = False
                details = f"Status: {response.status_code}"
                
            self.log_test_result("Feature Engineering", success, details, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Feature Engineering", False, f"Error: {str(e)}", duration)
    
    def test_predictions(self):
        """Test prediction functionality with various scenarios"""
        print("\n🎯 Testing Predictions...")
        
        test_cases = [
            {
                "name": "High-Risk Customer",
                "expected_high_risk": True,
                "data": {
                    "tenure": 1,
                    "MonthlyCharges": 85.0,
                    "TotalCharges": 85,
                    "gender": "Female",
                    "SeniorCitizen": 1,
                    "Partner": "No",
                    "Dependents": "No",
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check"
                }
            },
            {
                "name": "Low-Risk Customer",
                "expected_high_risk": False,
                "data": {
                    "tenure": 60,
                    "MonthlyCharges": 45.0,
                    "TotalCharges": 2700,
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "Yes",
                    "PhoneService": "Yes",
                    "MultipleLines": "Yes",
                    "InternetService": "DSL",
                    "OnlineSecurity": "Yes",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "Yes",
                    "TechSupport": "Yes",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Two year",
                    "PaperlessBilling": "No",
                    "PaymentMethod": "Bank transfer (automatic)"
                }
            },
            {
                "name": "Edge Case - Zero Tenure",
                "expected_high_risk": True,
                "data": {
                    "tenure": 0,
                    "MonthlyCharges": 75.0,
                    "TotalCharges": 0,
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "No",
                    "Dependents": "No",
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check"
                }
            }
        ]
        
        for test_case in test_cases:
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.api_base}/predict",
                    json=test_case["data"],
                    headers={"Content-Type": "application/json"},
                    timeout=15
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    prediction = response.json()
                    churn_prob = prediction.get("churn_probability", 0)
                    churn_pred = prediction.get("churn_prediction", False)
                    top_factors = prediction.get("top_factors", [])
                    
                    # Validate response structure
                    valid_structure = (
                        0 <= churn_prob <= 1 and
                        isinstance(churn_pred, bool) and
                        isinstance(top_factors, list) and
                        len(top_factors) >= 1
                    )
                    
                    # Check if prediction aligns with expectation
                    risk_aligned = (
                        (test_case["expected_high_risk"] and churn_prob > 0.5) or
                        (not test_case["expected_high_risk"] and churn_prob <= 0.5)
                    )
                    
                    success = valid_structure and risk_aligned
                    details = f"Prob: {churn_prob:.3f}, Pred: {churn_pred}, Factors: {len(top_factors)}"
                    
                    if not risk_aligned:
                        details += f" (Expected {'high' if test_case['expected_high_risk'] else 'low'} risk)"
                        
                else:
                    success = False
                    details = f"Status: {response.status_code}, Error: {response.text[:100]}"
                    
                self.log_test_result(test_case["name"], success, details, duration)
                
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(test_case["name"], False, f"Error: {str(e)}", duration)
    
    def test_error_handling(self):
        """Test API error handling"""
        print("\n🚨 Testing Error Handling...")
        
        # Test invalid JSON
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_base}/predict",
                data="invalid json",
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            duration = time.time() - start_time
            success = response.status_code in [400, 422]  # Expecting validation error
            details = f"Status: {response.status_code}"
            self.log_test_result("Invalid JSON Handling", success, details, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Invalid JSON Handling", False, f"Error: {str(e)}", duration)
        
        # Test missing required fields
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_base}/predict",
                json={"tenure": 12},  # Missing many required fields
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            duration = time.time() - start_time
            success = response.status_code == 422  # Expecting validation error
            details = f"Status: {response.status_code}"
            self.log_test_result("Missing Fields Handling", success, details, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Missing Fields Handling", False, f"Error: {str(e)}", duration)
        
        # Test invalid data types
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_base}/predict",
                json={
                    "tenure": "invalid",  # Should be int
                    "MonthlyCharges": 70.5,
                    "TotalCharges": 846,
                    "gender": "Male",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "PhoneService": "Yes",
                    "InternetService": "Fiber optic",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check"
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            duration = time.time() - start_time
            success = response.status_code == 422  # Expecting validation error
            details = f"Status: {response.status_code}"
            self.log_test_result("Invalid Data Types", success, details, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Invalid Data Types", False, f"Error: {str(e)}", duration)
    
    def test_performance(self):
        """Test API performance and load handling"""
        print("\n⚡ Testing Performance...")
        
        # Standard test data
        test_data = {
            "tenure": 12,
            "MonthlyCharges": 70.5,
            "TotalCharges": 846,
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check"
        }
        
        # Performance test
        response_times = []
        success_count = 0
        num_requests = 20
        
        print(f"   Running {num_requests} consecutive requests...")
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base}/predict",
                    json=test_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if response.status_code == 200:
                    success_count += 1
                    
            except Exception as e:
                print(f"   Request {i+1} failed: {str(e)}")
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            success_rate = success_count / num_requests
            
            # Performance criteria
            performance_ok = (
                avg_response_time < 2.0 and  # Average under 2 seconds
                max_response_time < 5.0 and  # Max under 5 seconds
                success_rate >= 0.95         # 95% success rate
            )
            
            details = f"Avg: {avg_response_time:.3f}s, Max: {max_response_time:.3f}s, Min: {min_response_time:.3f}s, Success: {success_rate:.1%}"
            self.log_test_result("Performance Test", performance_ok, details)
        else:
            self.log_test_result("Performance Test", False, "No successful requests")
    
    def test_mlflow_connectivity(self):
        """Test MLflow connectivity and functionality"""
        print("\n📊 Testing MLflow...")
        
        # Test MLflow UI accessibility
        start_time = time.time()
        try:
            response = requests.get(f"{self.mlflow_base}/", timeout=10)
            duration = time.time() - start_time
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            self.log_test_result("MLflow UI Access", success, details, duration)
            
            if success:
                # Test experiments endpoint
                try:
                    exp_response = requests.get(f"{self.mlflow_base}/api/2.0/mlflow/experiments/search", timeout=10)
                    if exp_response.status_code == 200:
                        experiments = exp_response.json().get("experiments", [])
                        exp_count = len(experiments)
                        has_experiments = exp_count > 0
                        details = f"Found {exp_count} experiments"
                        self.log_test_result("MLflow Experiments", has_experiments, details)
                    else:
                        self.log_test_result("MLflow Experiments", False, f"API Status: {exp_response.status_code}")
                except Exception as e:
                    self.log_test_result("MLflow Experiments", False, f"Error: {str(e)}")
                    
        except requests.exceptions.ConnectionError:
            duration = time.time() - start_time
            self.log_test_result("MLflow UI Access", False, "Connection refused - MLflow not running?", duration)
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("MLflow UI Access", False, f"Error: {str(e)}", duration)
    
    def test_data_consistency(self):
        """Test that predictions are consistent and reasonable"""
        print("\n🔍 Testing Data Consistency...")
        
        # Test same input gives same output
        test_data = {
            "tenure": 24,
            "MonthlyCharges": 65.0,
            "TotalCharges": 1560,
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Credit card (automatic)"
        }
        
        # Make 3 identical requests
        predictions = []
        for i in range(3):
            try:
                response = requests.post(
                    f"{self.api_base}/predict",
                    json=test_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                if response.status_code == 200:
                    predictions.append(response.json()["churn_probability"])
            except Exception as e:
                pass
        
        if len(predictions) >= 2:
            # Check consistency (small tolerance for floating point differences)
            consistent = all(abs(p - predictions[0]) < 1e-10 for p in predictions)
            details = f"Predictions: {predictions[:3]}"
            self.log_test_result("Prediction Consistency", consistent, details)
        else:
            self.log_test_result("Prediction Consistency", False, "Could not get multiple predictions")
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        total_time = time.time() - self.start_time
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        print(f"Total Duration: {total_time:.2f} seconds")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS ({failed_tests}):")
            print("-" * 50)
            for result in self.test_results:
                if not result["success"]:
                    print(f"• {result['test']}")
                    if result["details"]:
                        print(f"  └─ {result['details']}")
        
        print(f"\n⏱️  PERFORMANCE SUMMARY:")
        print("-" * 50)
        test_times = [r["duration"] for r in self.test_results if r["duration"] > 0]
        if test_times:
            print(f"Average Test Duration: {sum(test_times)/len(test_times):.3f}s")
            print(f"Slowest Test: {max(test_times):.3f}s")
            print(f"Fastest Test: {min(test_times):.3f}s")
        
        # Save report to file
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "success_rate": passed_tests/total_tests,
                    "total_duration": total_time,
                    "timestamp": datetime.now().isoformat()
                },
                "results": self.test_results
            }, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        return passed_tests == total_tests
    
    def run_all_tests(self):
        """Run comprehensive system tests"""
        print("🚀 STARTING COMPREHENSIVE CHURN PREDICTION SYSTEM TESTS")
        print("=" * 80)
        print(f"Test Suite Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target API: {self.api_base}")
        print(f"Target MLflow: {self.mlflow_base}")
        
        # Run all test suites
        self.test_docker_containers()
        self.test_api_connectivity()
        self.test_model_endpoints()
        self.test_predictions()
        self.test_error_handling()
        self.test_performance()
        self.test_mlflow_connectivity()
        self.test_data_consistency()
        
        # Generate final report
        all_passed = self.generate_report()
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! System is ready for production.")
        else:
            print("\n⚠️  Some tests failed. Please review the issues above.")
        
        return all_passed

def main():
    """Main function to run system tests"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("System Test Script for Churn Prediction API")
            print("Usage: python test_system.py [options]")
            print("Options:")
            print("  --help    Show this help message")
            print("  --quick   Run only essential tests (faster)")
            return
        elif sys.argv[1] == "--quick":
            print("Running quick tests only...")
            # Implement quick test mode if needed
    
    # Check if API is accessible before starting
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not accessible. Make sure the API is running:")
            print("   docker-compose up -d")
            print("   or")
            print("   uvicorn app.main:app --host 0.0.0.0 --port 8000")
            return False
    except Exception:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        return False
    
    # Run tests
    tester = SystemTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()