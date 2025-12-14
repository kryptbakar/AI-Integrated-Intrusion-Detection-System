"""
Google Drive File Downloader for Streamlit
Handles downloading large files from Google Drive with progress tracking
Uses gdown library for robust handling of virus scan warnings
"""

import os
import streamlit as st
from typing import Optional
import gdown


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


def download_from_google_drive_gdown(
    file_id: str,
    destination: str,
    show_progress: bool = True
) -> bool:
    """
    Download file from Google Drive using gdown (handles large files properly)
    
    Args:
        file_id: Google Drive file ID
        destination: Local path to save file
        show_progress: Whether to show progress
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Construct the download URL
        url = f"https://drive.google.com/uc?id={file_id}"
        
        if show_progress:
            st.info(f"📥 Starting download from Google Drive...")
            st.info(f"🔗 URL: {url}")
        
        # Use gdown to download (it handles virus scan warnings automatically)
        # fuzzy=True allows it to work even if direct download fails
        output = gdown.download(
            url=url,
            output=destination,
            quiet=False,
            fuzzy=True
        )
        
        # Check if download was successful
        if output is None:
            if show_progress:
                st.error("❌ gdown returned None - download failed")
            return False
        
        # Verify file exists and has content
        if not os.path.exists(destination):
            if show_progress:
                st.error("❌ File was not created at destination")
            return False
        
        file_size = os.path.getsize(destination)
        if file_size == 0:
            if show_progress:
                st.error("❌ Downloaded file is empty (0 bytes)")
            return False
        
        return True
        
    except Exception as e:
        if show_progress:
            st.error(f"❌ Download exception: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        return False


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def download_and_cache_dataset(
    google_drive_url: str,
    filename: str = "merged_output.csv"
) -> Optional[str]:
    """
    Download dataset from Google Drive and cache it
    Uses gdown library for robust large file handling
    
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
        st.error(f"Received: {google_drive_url}")
        st.info("💡 Expected format: https://drive.google.com/file/d/FILE_ID/view")
        return None
    
    st.info(f"🔍 Extracted File ID: `{file_id}`")
    
    # Create temp directory if it doesn't exist
    temp_dir = "/tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    destination = os.path.join(temp_dir, filename)
    
    # Check if already exists and is valid
    if os.path.exists(destination):
        file_size_mb = os.path.getsize(destination) / (1024 * 1024)
        if file_size_mb > 10:  # Only use cached if > 10MB (likely valid)
            st.success(f"✅ Using cached dataset ({file_size_mb:.1f} MB)")
            return destination
        else:
            # Remove invalid cached file
            os.remove(destination)
            st.warning(f"⚠️ Removed invalid cached file ({file_size_mb:.3f} MB)")
    
    # Download the file
    st.info("📥 Downloading dataset from Google Drive...")
    st.warning("⏳ Large files may take 2-5 minutes. This only happens once!")
    st.info("🔄 Using gdown library - handles virus scan warnings automatically")
    
    # Show the links for debugging
    with st.expander("🔍 Download Details"):
        st.code(f"File ID: {file_id}")
        st.code(f"Destination: {destination}")
        st.code(f"Direct link: https://drive.google.com/uc?id={file_id}")
        st.markdown("**Note:** gdown will handle Google's virus scan confirmation automatically")
    
    # Create a progress placeholder
    progress_placeholder = st.empty()
    with progress_placeholder:
        with st.spinner("Downloading... Please wait..."):
            success = download_from_google_drive_gdown(file_id, destination, show_progress=True)
    
    if success and os.path.exists(destination):
        file_size_mb = os.path.getsize(destination) / (1024 * 1024)
        
        if file_size_mb < 10:
            st.error(f"❌ Download produced suspiciously small file ({file_size_mb:.3f} MB)")
            st.error("**Possible issues:**")
            st.error("1. File might not be shared as 'Anyone with the link'")
            st.error("2. File ID might be incorrect")
            st.error("3. File might have been deleted from Google Drive")
            
            with st.expander("🔧 Troubleshooting Steps"):
                st.markdown("""
                ### How to fix:
                
                1. **Verify sharing settings:**
                   - Go to Google Drive
                   - Right-click your file
                   - Click "Share"
                   - Under "General access" select **"Anyone with the link"**
                   - Make sure it says **"Viewer"** (not "Restricted")
                   - Click "Copy link"
                
                2. **Test the link:**
                   - Open an **incognito/private browser window**
                   - Paste your Google Drive link
                   - You should see the file preview WITHOUT being asked to sign in
                   - If it asks to "Request access" → sharing is not set correctly
                
                3. **Check file ID:**
                   - Your link: `https://drive.google.com/file/d/FILE_ID_HERE/view`
                   - The FILE_ID should be a long random string
                   - Paste just the FILE_ID or the full link
                
                4. **Alternative: Use direct file ID**
                   - Instead of pasting full URL, try pasting just the FILE_ID
                   - Example: `1a2b3c4d5e6f7g8h9i0j`
                """)
            
            # Remove the invalid file
            if os.path.exists(destination):
                os.remove(destination)
            
            return None
        
        st.success(f"✅ Dataset downloaded successfully! ({file_size_mb:.1f} MB)")
        st.balloons()
        return destination
    else:
        st.error("❌ Download failed!")
        st.error("**Common causes:**")
        st.markdown("- ❌ File is not shared as 'Anyone with the link'")
        st.markdown("- ❌ File ID is incorrect") 
        st.markdown("- ❌ File was deleted from Google Drive")
        st.markdown("- ❌ Internet connection issue")
        
        with st.expander("📋 Step-by-step fix"):
            st.markdown("""
            ### Share your file correctly:
            
            1. Go to [Google Drive](https://drive.google.com)
            2. Find your `merged_output.csv` file
            3. Right-click → **Share**
            4. Click **"Change to anyone with the link"**
            5. Make sure it says **"Anyone with the link"** and **"Viewer"**
            6. Click **"Copy link"**
            7. Paste here in Streamlit
            
            ### Verify it's working:
            - Open link in incognito browser
            - Should show file preview immediately
            - Should NOT ask you to sign in or request access
            """)
        
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