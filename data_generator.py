import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

class StudentDataGenerator:
    """
    Generates dummy student data that simulates data from a spreadsheet.
    This includes academic performance, attendance, and demographic information.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Define realistic data pools
        self.first_names = [
            "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
            "Isabella", "William", "Mia", "James", "Charlotte", "Benjamin", "Amelia",
            "Lucas", "Harper", "Henry", "Evelyn", "Alexander", "Abigail", "Michael"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
        ]
        
        self.subjects = [
            "Mathematics", "English", "Science", "History", "Art", "Physical Education",
            "Computer Science", "Foreign Language", "Music", "Chemistry", "Physics", "Biology"
        ]
        
        self.grade_levels = ["9th", "10th", "11th", "12th"]
        
    def generate_student_data(self, num_students: int = 100) -> pd.DataFrame:
        """
        Generate a DataFrame with student information similar to spreadsheet data.
        
        Args:
            num_students: Number of students to generate
            
        Returns:
            DataFrame with student data
        """
        students = []
        
        for i in range(num_students):
            student = {
                "student_id": f"STU{i+1:04d}",
                "first_name": random.choice(self.first_names),
                "last_name": random.choice(self.last_names),
                "grade_level": random.choice(self.grade_levels),
                "age": random.randint(14, 18),
                "gpa": round(random.uniform(2.0, 4.0), 2),
                "attendance_rate": round(random.uniform(75, 100), 1),
                "assignments_completed": random.randint(15, 30),
                "assignments_total": 30,
                "test_scores_avg": round(random.uniform(60, 100), 1),
                "participation_score": random.randint(1, 10),
                "homework_completion_rate": round(random.uniform(70, 100), 1),
                "extracurricular_activities": random.randint(0, 5),
                "study_hours_per_week": random.randint(5, 25),
                "parent_education_level": random.choice(["High School", "Bachelor's", "Master's", "PhD"]),
                "socioeconomic_status": random.choice(["Low", "Middle", "High"]),
                "previous_grade_performance": round(random.uniform(2.0, 4.0), 2),
                "learning_style": random.choice(["Visual", "Auditory", "Kinesthetic", "Mixed"]),
                "special_needs": random.choice([True, False]) if random.random() < 0.15 else False,
                "english_as_second_language": random.choice([True, False]) if random.random() < 0.20 else False
            }
            
            # Add some correlation between related fields
            if student["gpa"] > 3.5:
                student["test_scores_avg"] = max(student["test_scores_avg"], 80)
                student["attendance_rate"] = max(student["attendance_rate"], 90)
            
            students.append(student)
        
        return pd.DataFrame(students)
    
    def generate_subject_grades(self, student_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate subject-specific grades for each student.
        
        Args:
            student_df: DataFrame with student basic information
            
        Returns:
            DataFrame with subject grades
        """
        subject_grades = []
        
        for _, student in student_df.iterrows():
            base_performance = student["gpa"] / 4.0  # Normalize to 0-1
            
            for subject in self.subjects:
                # Add some variation around the base performance
                subject_performance = base_performance + random.uniform(-0.2, 0.2)
                subject_performance = max(0.0, min(1.0, subject_performance))  # Clamp to 0-1
                
                grade_record = {
                    "student_id": student["student_id"],
                    "subject": subject,
                    "current_grade": round(subject_performance * 100, 1),
                    "quarter1_grade": round((subject_performance + random.uniform(-0.1, 0.1)) * 100, 1),
                    "quarter2_grade": round((subject_performance + random.uniform(-0.1, 0.1)) * 100, 1),
                    "quarter3_grade": round((subject_performance + random.uniform(-0.1, 0.1)) * 100, 1),
                    "midterm_exam": round((subject_performance + random.uniform(-0.15, 0.15)) * 100, 1),
                    "final_exam": round((subject_performance + random.uniform(-0.15, 0.15)) * 100, 1),
                    "participation": random.randint(1, 10),
                    "homework_avg": round((subject_performance + random.uniform(-0.1, 0.1)) * 100, 1)
                }
                
                subject_grades.append(grade_record)
        
        return pd.DataFrame(subject_grades)
    
    def save_to_excel(self, student_df: pd.DataFrame, subject_df: pd.DataFrame, filename: str = "student_data.xlsx"):
        """
        Save the generated data to an Excel file with multiple sheets.
        
        Args:
            student_df: Student information DataFrame
            subject_df: Subject grades DataFrame
            filename: Output filename
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            student_df.to_excel(writer, sheet_name='Students', index=False)
            subject_df.to_excel(writer, sheet_name='Grades', index=False)
        
        print(f"Data saved to {filename}")
    
    def get_student_summary(self, student_id: str, student_df: pd.DataFrame, subject_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a comprehensive summary for a specific student.
        
        Args:
            student_id: The student ID to look up
            student_df: Student information DataFrame
            subject_df: Subject grades DataFrame
            
        Returns:
            Dictionary with student summary
        """
        student_info = student_df[student_df["student_id"] == student_id].iloc[0].to_dict()
        student_grades = subject_df[subject_df["student_id"] == student_id]
        
        summary = {
            "basic_info": student_info,
            "grades_by_subject": student_grades.to_dict('records'),
            "overall_performance": {
                "average_current_grade": student_grades["current_grade"].mean(),
                "strongest_subject": student_grades.loc[student_grades["current_grade"].idxmax(), "subject"],
                "weakest_subject": student_grades.loc[student_grades["current_grade"].idxmin(), "subject"],
                "grade_trend": "improving" if student_grades["final_exam"].mean() > student_grades["quarter1_grade"].mean() else "declining"
            }
        }
        
        return summary


if __name__ == "__main__":
    # Example usage
    generator = StudentDataGenerator()
    
    # Generate student data
    students = generator.generate_student_data(50)
    grades = generator.generate_subject_grades(students)
    
    # Save to Excel file
    generator.save_to_excel(students, grades)
    
    # Print sample data
    print("Sample Student Data:")
    print(students.head())
    print("\nSample Grade Data:")
    print(grades.head())
    
    # Get summary for first student
    sample_student = students.iloc[0]["student_id"]
    summary = generator.get_student_summary(sample_student, students, grades)
    print(f"\nSample Summary for {sample_student}:")
    print(f"GPA: {summary['basic_info']['gpa']}")
    print(f"Average Current Grade: {summary['overall_performance']['average_current_grade']:.1f}")
    print(f"Strongest Subject: {summary['overall_performance']['strongest_subject']}")