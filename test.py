import os

# Path to the directory where you think the model might be saved
search_dir = 'C:/Users/khush/Desktop'  # Adjust the directory as needed

# Search for the file
for root, dirs, files in os.walk(search_dir):
    if 'action.keras' in files:
        print(os.path.join(root, 'action.keras'))
