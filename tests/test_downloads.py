from unittest.mock import patch, call
import pytest

from coastseg import downloads

def test_authenticate_and_initialize_max_attempts():
    with patch('coastseg.downloads.ee.Authenticate') as mock_authenticate, \
         patch('coastseg.downloads.ee.Initialize') as mock_initialize:
        
        # Mock an exception for all initialize attempts
        mock_initialize.side_effect = Exception("Credentials file not found")
        
        with pytest.raises(Exception) as excinfo:
            downloads.authenticate_and_initialize(print_mode=True, force=False, auth_args={}, kwargs={})
        
        assert "Failed to initialize Google Earth Engine after 2 attempts" in str(excinfo.value)
        assert mock_authenticate.call_count == 2
        assert mock_initialize.call_count == 2


def test_authenticate_and_initialize_success():
    with patch('coastseg.downloads.ee.Authenticate') as mock_authenticate, \
         patch('coastseg.downloads.ee.Initialize') as mock_initialize:
        
        # Mock successful initialization
        mock_initialize.return_value = None
        
        downloads.authenticate_and_initialize(print_mode=True, force=False, auth_args={}, kwargs={})

        mock_authenticate.assert_called_once() # this will call once becase ee.credentials is None
        mock_initialize.assert_called_once()

def test_authenticate_and_initialize_force_auth():
    with patch('coastseg.downloads.ee.Authenticate') as mock_authenticate, \
         patch('coastseg.downloads.ee.Initialize') as mock_initialize:
        
        # Mock successful initialization
        mock_initialize.return_value = None
        
        downloads.authenticate_and_initialize(print_mode=True, force=True, auth_args={}, kwargs={})

        mock_authenticate.assert_called_once_with(force=True)
        mock_initialize.assert_called_once()

def test_authenticate_and_initialize_retry():
    with patch('coastseg.downloads.ee.Authenticate') as mock_authenticate, \
         patch('coastseg.downloads.ee.Initialize') as mock_initialize:
        
        # Mock an exception on first initialize, then success
        mock_initialize.side_effect = [Exception("Credentials file not found"), None]

        downloads.authenticate_and_initialize(print_mode=True, force=False, auth_args={}, kwargs={})

        assert mock_authenticate.call_count == 2
        assert mock_initialize.call_count == 2
