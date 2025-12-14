from typing import TypedDict

### typeddict_demo ---> This is an example of how to use TypedDict in Python to define a structured data type.
### It is a method to get the structured output from LLMs.

class MovieInfo(TypedDict):
    title: str
    director: str
    year: int
    genre: str

new_movie: MovieInfo = {
    "title": "Inception",
    "director": "Christopher Nolan",
    "year": 2010,
    "genre": "Science Fiction"
}

print(new_movie)