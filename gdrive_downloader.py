"""
Google Drive File Downloader for Streamlit
Handles downloading large files from Google Drive with progress tracking
"""

import os
import requests
import streamlit as st
from typing import Optional
import time


def extract_file_id(google_drive_url: str) -> Optional[str]:
    """
    Extract file ID from various Google Drive URL formats
    
    Supports:
    - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    - https://drive.google.com/open?id=FILE_ID
    - https://drive.google.com/uc?id=FILE_ID
    - Direct FILE_ID
    """
    if '/' in google_drive_url or '?' in google_drive_url:
        # Extract from URL
        if '/file/d/' in google_drive_url:
            file_id = google_drive_url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in google_drive_url:
            file_id = google_drive_url.split('id=')[1].split('&')[0]
        else:
            return None
    else:
        # Assume it's already a file ID
        file_id = google_drive_url.strip()
    
    return file_id


def get_file_size(file_id: str) -> Optional[int]:
    """Get file size from Google Drive without downloading"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        response = requests.head(url, allow_redirects=True)
        if 'Content-Length' in response.headers:
            return int(response.headers['Content-Length'])
    except:
        pass
    
    return None


def download_from_google_drive(
    file_id: str,
    destination: str,
    show_progress: bool = True
) -> bool:
    """
    Download file from Google Drive with progress tracking
    
    Args:
        file_id: Google Drive file ID
        destination: Local path to save file
        show_progress: Whether to show Streamlit progress bar
        
    Returns:
        True if successful, False otherwise
    """
    
    # Google Drive download URL
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        # Start session
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Handle confirmation token for large files
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)
        
        # Check if successful
        if response.status_code != 200:
            return False
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Progress tracking
        if show_progress and total_size > 0:
            progress_bar = st.progress(0)
            status_text = st.empty()
            size_mb = total_size / (1024 * 1024)
            status_text.text(f"📥 Downloading dataset ({size_mb:.1f} MB)...")
        
        # Download in chunks
        chunk_size = 32768  # 32KB chunks
        downloaded = 0
        start_time = time.time()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    if show_progress and total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        
                        # Calculate speed
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            speed_mbps = (downloaded / (1024 * 1024)) / elapsed
                            downloaded_mb = downloaded / (1024 * 1024)
                            status_text.text(
                                f"📥 Downloading: {downloaded_mb:.1f}/{size_mb:.1f} MB "
                                f"({speed_mbps:.1f} MB/s)"
                            )
        
        # Clean up progress indicators
        if show_progress:
            progress_bar.empty()
            status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return False


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def download_and_cache_dataset(
    google_drive_url: str,
    filename: str = "merged_output.csv"
) -> Optional[str]:
    """
    Download dataset from Google Drive and cache it
    
    Args:
        google_drive_url: Google Drive share link or file ID
        filename: Name to save the file as
        
    Returns:
        Path to downloaded file, or None if failed
    """
    
    # Extract file ID
    file_id = extract_file_id(google_drive_url)
    
    if not file_id:
        st.error("❌ Invalid Google Drive URL or file ID")
        return None
    
    # Create temp directory if it doesn't exist
    temp_dir = "/tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    destination = os.path.join(temp_dir, filename)
    
    # Check if already exists
    if os.path.exists(destination):
        file_size_mb = os.path.getsize(destination) / (1024 * 1024)
        st.success(f"✅ Using cached dataset ({file_size_mb:.1f} MB)")
        return destination
    
    # Download the file
    st.info("📥 Downloading dataset from Google Drive...")
    st.warning("⏳ First download may take 1-2 minutes. Subsequent loads will be instant!")
    
    success = download_from_google_drive(file_id, destination, show_progress=True)
    
    if success and os.path.exists(destination):
        file_size_mb = os.path.getsize(destination) / (1024 * 1024)
        st.success(f"✅ Dataset downloaded successfully! ({file_size_mb:.1f} MB)")
        return destination
    else:
        st.error("❌ Download failed. Please check your Google Drive link.")
        return None


def get_google_drive_shareable_link_instructions():
    """Return instructions for creating a shareable Google Drive link"""
    
    instructions = """
    ### 📋 How to Get Google Drive Shareable Link:
    
    1. **Upload your file to Google Drive**
       - Go to drive.google.com
       - Click "New" → "File upload"
       - Upload `merged_output.csv` (700MB)
    
    2. **Get shareable link:**
       - Right-click the uploaded file
       - Click "Share" or "Get link"
       - Set to **"Anyone with the link"** can **view**
       - Click "Copy link"
    
    3. **Paste the link in Streamlit sidebar**
       - Example: `https://drive.google.com/file/d/1a2B3c4D5e6F7g8H9/view?usp=sharing`
       - Or just the file ID: `1a2B3c4D5e6F7g8H9`
    
    **Both formats work!** ✅
    """
    
    return instructions