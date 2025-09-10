import requests

# Test TMDB API key
def test_tmdb_api():
    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    movie_id = 19995  # Avatar movie ID
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Movie Title: {data.get('title', 'N/A')}")
            print(f"Overview: {data.get('overview', 'N/A')[:100]}...")
            
            if 'poster_path' in data:
                poster_url = "https://image.tmdb.org/t/p/w500" + data['poster_path']
                print(f"Poster URL: {poster_url}")
            else:
                print("No poster found")
                
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_tmdb_api()
