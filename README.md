# Student Performance Prediction with Mock AWS SageMaker LLM

This project demonstrates how to use AWS SageMaker for hosting and using an LLM to predict student performance. The application generates dummy student data (simulating spreadsheet data) and uses a mock SageMaker LLM service to make various predictions about student outcomes.

## Features

- **Dummy Data Generation**: Creates realistic student data with demographics, grades, attendance, and performance metrics
- **Mock SageMaker LLM**: Simulates AWS SageMaker endpoint for student performance predictions
- **Multiple Prediction Types**:
  - Graduation probability prediction
  - At-risk student assessment
  - Next semester GPA prediction
  - Personalized improvement recommendations
- **Batch Processing**: Support for processing multiple students simultaneously
- **Data Export**: Save results to Excel and JSON formats
- **Interactive Demo**: Complete workflow demonstration

## Project Structure

```
├── main.py                 # Main application entry point
├── data_generator.py       # Student data generation module
├── sagemaker_mock.py       # Mock SageMaker LLM service
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python main.py --help
   ```

## Quick Start

### Run the Interactive Demo

The easiest way to see the application in action is to run the interactive demo:

```bash
python main.py --demo
```

This will:
- Generate dummy student data (20 students)
- Deploy the mock SageMaker LLM model
- Show detailed analysis for a sample student
- Make various types of predictions
- Perform batch predictions
- Save results to files

### Basic Usage Examples

1. **Generate data and analyze a specific student**:
   ```bash
   python main.py --student-id STU0001
   ```

2. **Run batch predictions for multiple students**:
   ```bash
   python main.py --batch 10 --prediction at_risk_assessment
   ```

3. **Generate more students and make GPA predictions**:
   ```bash
   python main.py --students 50 --prediction next_semester_gpa
   ```

## Command Line Options

```
usage: main.py [-h] [--demo] [--students STUDENTS] [--student-id STUDENT_ID]
               [--prediction {graduation_probability,at_risk_assessment,improvement_recommendations,next_semester_gpa}]
               [--batch BATCH]

optional arguments:
  -h, --help            show this help message and exit
  --demo                Run interactive demo
  --students STUDENTS   Number of students to generate (default: 100)
  --student-id STUDENT_ID
                        Specific student ID to analyze
  --prediction {graduation_probability,at_risk_assessment,improvement_recommendations,next_semester_gpa}
                        Type of prediction to make (default: graduation_probability)
  --batch BATCH         Number of students for batch prediction
```

## Prediction Types

### 1. Graduation Probability
Predicts the likelihood of a student graduating on time based on:
- Current GPA
- Attendance rate
- Test scores
- Participation levels
- Homework completion rate

**Example Output**:
```
PREDICTION RESULTS: GRADUATION_PROBABILITY
============================================================
Student ID: STU0001
Graduation Probability: 78.3%
Confidence Score: 87.2%
Risk Level: Medium
Risk Factors: Poor attendance, Low test scores
```

### 2. At-Risk Assessment
Comprehensive risk evaluation identifying students who may need intervention:
- Risk scoring based on multiple factors
- Specific risk factor identification
- Actionable recommendations

**Example Output**:
```
PREDICTION RESULTS: AT_RISK_ASSESSMENT
============================================================
Student ID: STU0001
Risk Level: High
Risk Score: 35
Priority: Close monitoring needed
Risk Factors:
  - Low GPA (< 2.5)
  - Below average attendance
Recommendations:
  - Academic support recommended
  - Monitor attendance closely
```

### 3. Improvement Recommendations
Personalized recommendations based on student profile:
- Academic strategies
- Behavioral improvements
- Support services
- Extracurricular guidance

### 4. Next Semester GPA Prediction
Forecasts academic performance for the following semester:
- Trend analysis
- Factor-based adjustments
- Confidence scoring

## Data Structure

### Student Data Fields
The application generates comprehensive student profiles including:

- **Demographics**: Name, age, grade level
- **Academic Performance**: GPA, test scores, grades by subject
- **Behavioral Metrics**: Attendance rate, participation scores
- **Study Habits**: Study hours per week, homework completion
- **Background**: Socioeconomic status, parent education, learning style
- **Special Considerations**: Special needs, ESL status

### Generated Files

The application creates several output files:

1. **`student_data_YYYYMMDD_HHMMSS.xlsx`**: Complete student and grade data in Excel format
2. **`prediction_results_YYYYMMDD_HHMMSS.json`**: Prediction results in JSON format

## Mock SageMaker LLM Details

The mock SageMaker service simulates real AWS SageMaker behavior:

- **Deployment simulation**: Mimics model deployment with instance types
- **Endpoint management**: Tracks deployment status and endpoints
- **Realistic processing times**: Simulates LLM inference latency
- **Intelligent predictions**: Uses weighted algorithms for realistic results
- **Batch processing**: Supports multiple students simultaneously

### Supported Endpoints

- Graduation probability prediction
- Risk assessment analysis
- GPA forecasting
- Recommendation generation

## Real AWS SageMaker Integration

To adapt this code for real AWS SageMaker:

1. **Replace `MockSageMakerLLM`** with actual SageMaker client:
   ```python
   import boto3
   sagemaker_runtime = boto3.client('sagemaker-runtime')
   ```

2. **Update prediction methods** to use real endpoints:
   ```python
   response = sagemaker_runtime.invoke_endpoint(
       EndpointName='your-endpoint-name',
       ContentType='application/json',
       Body=json.dumps(student_data)
   )
   ```

3. **Add AWS credentials** and region configuration
4. **Deploy your actual LLM model** to SageMaker

## Educational Use Cases

This application demonstrates several educational analytics scenarios:

1. **Early Warning Systems**: Identify at-risk students before they fail
2. **Personalized Learning**: Tailor recommendations to individual learning styles
3. **Resource Allocation**: Prioritize intervention efforts based on risk scores
4. **Progress Monitoring**: Track student development over time
5. **Predictive Analytics**: Forecast academic outcomes for planning

## Customization

### Adding New Prediction Types

To add new prediction capabilities:

1. **Extend `MockSageMakerLLM`** with new prediction methods
2. **Update `main.py`** to handle the new prediction type
3. **Add command line options** for the new prediction
4. **Update display methods** for formatted output

### Modifying Student Data

To customize the generated student data:

1. **Edit `data_generator.py`**:
   - Modify field lists (names, subjects, etc.)
   - Adjust data ranges and distributions
   - Add new calculated fields

2. **Update correlation rules** between related fields

### Real Data Integration

To use real spreadsheet data instead of generated data:

1. **Replace data generation** with data loading:
   ```python
   self.students_df = pd.read_excel('real_student_data.xlsx')
   ```

2. **Ensure data format compatibility** with expected field names
3. **Add data validation** and cleaning steps

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **Permission errors**: Ensure write permissions for output files
3. **Memory issues**: Reduce the number of students for large datasets

### Debug Mode

Enable detailed logging by modifying the logging level in `main.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

This is a demonstration project showing how to integrate dummy data with mock SageMaker services. Feel free to:

- Add more sophisticated prediction algorithms
- Enhance the data generation with more realistic patterns
- Integrate with real AWS SageMaker endpoints
- Add visualization capabilities
- Implement real-time monitoring features

## License

This project is provided as-is for educational and demonstration purposes.

## Next Steps

To use this in a production environment:

1. **Deploy a real LLM** to AWS SageMaker
2. **Integrate with actual student information systems**
3. **Add authentication and authorization**
4. **Implement data privacy and security measures**
5. **Add monitoring and logging**
6. **Create a web interface** for easier interaction
7. **Set up automated scheduling** for regular predictions

---

**Note**: This application uses mock services and dummy data for demonstration purposes. In a real educational environment, ensure compliance with student privacy regulations (FERPA, GDPR, etc.) and implement appropriate security measures.