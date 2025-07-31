#!/usr/bin/env python3
"""
Main application for Student Performance Prediction using Mock AWS SageMaker LLM.

This application demonstrates how to:
1. Generate dummy student data (simulating spreadsheet data)
2. Deploy a mock SageMaker LLM endpoint
3. Make predictions for student performance
4. Display results in a user-friendly format

Author: AI Assistant
Date: 2024
"""

import sys
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime
import logging

# Import our custom modules
from data_generator import StudentDataGenerator
from sagemaker_mock import MockSageMakerLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class StudentPredictionApp:
    """
    Main application class that orchestrates the student prediction workflow.
    """
    
    def __init__(self):
        """Initialize the application components."""
        self.data_generator = StudentDataGenerator()
        self.llm_service = MockSageMakerLLM()
        self.students_df = None
        self.grades_df = None
        
        logger.info("Student Prediction Application initialized")
    
    def generate_student_data(self, num_students: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate dummy student data.
        
        Args:
            num_students: Number of students to generate
            
        Returns:
            Tuple of (students_df, grades_df)
        """
        logger.info(f"Generating data for {num_students} students...")
        
        self.students_df = self.data_generator.generate_student_data(num_students)
        self.grades_df = self.data_generator.generate_subject_grades(self.students_df)
        
        logger.info(f"Generated data for {len(self.students_df)} students with {len(self.grades_df)} grade records")
        
        return self.students_df, self.grades_df
    
    def deploy_llm_model(self, instance_type: str = "ml.m5.large") -> Dict[str, Any]:
        """
        Deploy the mock SageMaker LLM model.
        
        Args:
            instance_type: EC2 instance type for deployment
            
        Returns:
            Deployment information
        """
        logger.info("Deploying SageMaker LLM model...")
        deployment_info = self.llm_service.deploy_model(instance_type=instance_type)
        logger.info("Model deployed successfully")
        return deployment_info
    
    def predict_student_performance(self, student_id: str, prediction_type: str = "graduation_probability") -> Dict[str, Any]:
        """
        Make a prediction for a specific student.
        
        Args:
            student_id: ID of the student
            prediction_type: Type of prediction to make
            
        Returns:
            Prediction results
        """
        if self.students_df is None:
            raise ValueError("No student data available. Generate data first.")
        
        # Get student data
        student_row = self.students_df[self.students_df["student_id"] == student_id]
        if student_row.empty:
            raise ValueError(f"Student {student_id} not found")
        
        student_data = student_row.iloc[0].to_dict()
        
        # Get student grades for certain predictions
        student_grades = self.grades_df[self.grades_df["student_id"] == student_id].to_dict('records') if self.grades_df is not None else []
        
        logger.info(f"Making {prediction_type} prediction for student {student_id}")
        
        # Make prediction based on type
        if prediction_type == "graduation_probability":
            result = self.llm_service.predict_graduation_probability(student_data)
        elif prediction_type == "next_semester_gpa":
            result = self.llm_service.predict_next_semester_gpa(student_data, student_grades)
        elif prediction_type == "at_risk_assessment":
            result = self.llm_service.assess_at_risk_status(student_data)
        elif prediction_type == "improvement_recommendations":
            result = self.llm_service.generate_improvement_recommendations(student_data)
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")
        
        return result
    
    def batch_predict_students(self, prediction_type: str = "graduation_probability", max_students: int = 10) -> List[Dict[str, Any]]:
        """
        Make batch predictions for multiple students.
        
        Args:
            prediction_type: Type of prediction to make
            max_students: Maximum number of students to process
            
        Returns:
            List of prediction results
        """
        if self.students_df is None:
            raise ValueError("No student data available. Generate data first.")
        
        # Limit the number of students for demo purposes
        students_subset = self.students_df.head(max_students)
        students_data = students_subset.to_dict('records')
        
        logger.info(f"Making batch {prediction_type} predictions for {len(students_data)} students")
        
        results = self.llm_service.batch_predict(students_data, prediction_type)
        return results
    
    def display_student_summary(self, student_id: str):
        """
        Display a comprehensive summary for a student.
        
        Args:
            student_id: ID of the student
        """
        if self.students_df is None or self.grades_df is None:
            print("No student data available. Generate data first.")
            return
        
        summary = self.data_generator.get_student_summary(student_id, self.students_df, self.grades_df)
        
        print("\n" + "="*60)
        print(f"STUDENT SUMMARY: {student_id}")
        print("="*60)
        
        # Basic info
        basic_info = summary["basic_info"]
        print(f"Name: {basic_info['first_name']} {basic_info['last_name']}")
        print(f"Grade Level: {basic_info['grade_level']}")
        print(f"Age: {basic_info['age']}")
        print(f"GPA: {basic_info['gpa']}")
        print(f"Attendance Rate: {basic_info['attendance_rate']}%")
        print(f"Study Hours/Week: {basic_info['study_hours_per_week']}")
        print(f"Learning Style: {basic_info['learning_style']}")
        
        # Performance summary
        performance = summary["overall_performance"]
        print(f"\nOverall Performance:")
        print(f"  Average Current Grade: {performance['average_current_grade']:.1f}%")
        print(f"  Strongest Subject: {performance['strongest_subject']}")
        print(f"  Weakest Subject: {performance['weakest_subject']}")
        print(f"  Grade Trend: {performance['grade_trend']}")
    
    def display_prediction_results(self, results: Dict[str, Any]):
        """
        Display prediction results in a formatted way.
        
        Args:
            results: Prediction results dictionary
        """
        print("\n" + "="*60)
        print(f"PREDICTION RESULTS: {results['prediction_type'].upper()}")
        print("="*60)
        
        prediction_type = results["prediction_type"]
        
        if prediction_type == "graduation_probability":
            print(f"Student ID: {results['student_id']}")
            print(f"Graduation Probability: {results['graduation_probability']}%")
            print(f"Confidence Score: {results['confidence_score']}%")
            print(f"Risk Level: {results['risk_level']}")
            if results['risk_factors']:
                print(f"Risk Factors: {', '.join(results['risk_factors'])}")
        
        elif prediction_type == "next_semester_gpa":
            print(f"Student ID: {results['student_id']}")
            print(f"Current GPA: {results['current_gpa']}")
            print(f"Predicted GPA: {results['predicted_gpa']}")
            print(f"GPA Change: {results['gpa_change']:+.2f}")
            print(f"Trend: {results['trend']}")
            print(f"Confidence Score: {results['confidence_score']}%")
        
        elif prediction_type == "at_risk_assessment":
            print(f"Student ID: {results['student_id']}")
            print(f"Risk Level: {results['risk_level']}")
            print(f"Risk Score: {results['risk_score']}")
            print(f"Priority: {results['priority']}")
            if results['risk_factors']:
                print(f"Risk Factors:")
                for factor in results['risk_factors']:
                    print(f"  - {factor}")
            if results['recommendations']:
                print(f"Recommendations:")
                for rec in results['recommendations']:
                    print(f"  - {rec}")
        
        elif prediction_type == "improvement_recommendations":
            print(f"Student ID: {results['student_id']}")
            print(f"Priority Focus: {results['priority_focus']}")
            print(f"Implementation Timeline: {results['implementation_timeline']}")
            print(f"Confidence Score: {results['confidence_score']}%")
            
            recommendations = results['recommendations']
            for category, recs in recommendations.items():
                if recs:
                    print(f"\n{category.title()} Recommendations:")
                    for rec in recs:
                        print(f"  - {rec}")
        
        print(f"\nPrediction Timestamp: {results['prediction_timestamp']}")
        print(f"Model Version: {results['model_version']}")
    
    def save_results_to_file(self, results: List[Dict[str, Any]], filename: str):
        """
        Save prediction results to a JSON file.
        
        Args:
            results: List of prediction results
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filename}")
    
    def run_interactive_demo(self):
        """Run an interactive demonstration of the application."""
        print("\n" + "="*80)
        print("STUDENT PERFORMANCE PREDICTION DEMO")
        print("Powered by Mock AWS SageMaker LLM")
        print("="*80)
        
        # Step 1: Generate data
        print("\n1. Generating student data...")
        self.generate_student_data(num_students=20)
        print(f"✓ Generated data for {len(self.students_df)} students")
        
        # Step 2: Deploy model
        print("\n2. Deploying SageMaker LLM model...")
        deployment_info = self.deploy_llm_model()
        print(f"✓ Model deployed to endpoint: {deployment_info['endpoint_name']}")
        
        # Step 3: Show sample student
        sample_student_id = self.students_df.iloc[0]["student_id"]
        print(f"\n3. Sample student analysis...")
        self.display_student_summary(sample_student_id)
        
        # Step 4: Make predictions for sample student
        prediction_types = ["graduation_probability", "at_risk_assessment", "improvement_recommendations"]
        
        for pred_type in prediction_types:
            print(f"\n4.{prediction_types.index(pred_type) + 1}. Making {pred_type} prediction...")
            result = self.predict_student_performance(sample_student_id, pred_type)
            self.display_prediction_results(result)
        
        # Step 5: Batch predictions
        print(f"\n5. Running batch graduation probability predictions...")
        batch_results = self.batch_predict_students("graduation_probability", max_students=5)
        
        print(f"\nBatch Prediction Summary (5 students):")
        print("-" * 50)
        for result in batch_results:
            print(f"{result['student_id']}: {result['graduation_probability']}% ({result['risk_level']} risk)")
        
        # Step 6: Save results
        print(f"\n6. Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_results_to_file(batch_results, f"prediction_results_{timestamp}.json")
        
        # Save student data to Excel
        self.data_generator.save_to_excel(self.students_df, self.grades_df, f"student_data_{timestamp}.xlsx")
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nFiles generated:")
        print(f"- student_data_{timestamp}.xlsx (Student and grade data)")
        print(f"- prediction_results_{timestamp}.json (Prediction results)")
        print("\nThe application successfully demonstrated:")
        print("1. ✓ Dummy data generation (simulating spreadsheet data)")
        print("2. ✓ Mock SageMaker LLM deployment")
        print("3. ✓ Student performance predictions")
        print("4. ✓ Risk assessments and recommendations")
        print("5. ✓ Batch processing capabilities")


def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Student Performance Prediction using Mock AWS SageMaker LLM")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--students", type=int, default=100, help="Number of students to generate")
    parser.add_argument("--student-id", type=str, help="Specific student ID to analyze")
    parser.add_argument("--prediction", type=str, choices=["graduation_probability", "at_risk_assessment", "improvement_recommendations", "next_semester_gpa"], 
                       default="graduation_probability", help="Type of prediction to make")
    parser.add_argument("--batch", type=int, help="Number of students for batch prediction")
    
    args = parser.parse_args()
    
    app = StudentPredictionApp()
    
    try:
        if args.demo:
            # Run the interactive demo
            app.run_interactive_demo()
        else:
            # Generate data
            app.generate_student_data(num_students=args.students)
            
            # Deploy model
            app.deploy_llm_model()
            
            if args.student_id:
                # Single student prediction
                app.display_student_summary(args.student_id)
                result = app.predict_student_performance(args.student_id, args.prediction)
                app.display_prediction_results(result)
            elif args.batch:
                # Batch prediction
                results = app.batch_predict_students(args.prediction, max_students=args.batch)
                for result in results:
                    app.display_prediction_results(result)
                    print("-" * 60)
            else:
                # Default: show first student
                first_student = app.students_df.iloc[0]["student_id"]
                app.display_student_summary(first_student)
                result = app.predict_student_performance(first_student, args.prediction)
                app.display_prediction_results(result)
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()