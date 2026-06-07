from src.admin_storage import log_analysis_event, read_history

def test_log_analysis_event():
    assert log_analysis_event({"analysis_type": "test", "risk_level": "Low", "status": "success"}) is True
    rows = read_history(5)
    assert len(rows) >= 1

