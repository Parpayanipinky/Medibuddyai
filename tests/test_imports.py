def test_architecture_imports():
  from src.architecture import ReportAnalyzer, UserInterface, AdminAnalytics
  assert ReportAnalyzer is not None
  assert UserInterface is not None
  assert AdminAnalytics is not None

