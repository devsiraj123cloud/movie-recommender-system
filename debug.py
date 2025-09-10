import os
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))

# Try to create a test file
try:
    with open('test_file.txt', 'w') as f:
        f.write('test')
    print("Test file created successfully")
    print("Files after creating test file:", os.listdir('.'))
except Exception as e:
    print(f"Error creating test file: {e}")
