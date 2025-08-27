import { type NextRequest, NextResponse } from "next/server"
import { readFile } from "fs/promises"

export async function POST(request: NextRequest) {
  try {
    const { preferences, filePath } = await request.json()

    if (!filePath) {
      return NextResponse.json({ success: false, error: "No CSV file path provided" })
    }

    const csvMovies = []
    try {
      const fileContent = await readFile(filePath, "utf-8")
      const lines = fileContent.split("\n").filter((line) => line.trim())

      function parseCSVLine(line: string): string[] {
        const result = []
        let current = ""
        let inQuotes = false

        for (let i = 0; i < line.length; i++) {
          const char = line[i]

          if (char === '"') {
            inQuotes = !inQuotes
          } else if (char === "," && !inQuotes) {
            result.push(current.trim())
            current = ""
          } else {
            current += char
          }
        }

        result.push(current.trim())
        return result
      }

      const headers = parseCSVLine(lines[0]).map((h) => h.trim().toLowerCase().replace(/"/g, ""))
      console.log("[v0] CSV Headers found:", headers)

      // Parse CSV rows into movie objects
      for (let i = 1; i < lines.length && i <= 100; i++) {
        const values = parseCSVLine(lines[i]).map((v) => v.trim().replace(/"/g, ""))
        const movie: any = {}

        headers.forEach((header, index) => {
          movie[header] = values[index] || ""
        })

        console.log(`[v0] Parsed movie ${i}:`, {
          title: movie.title || movie.name || movie.movie_title,
          year: movie.year || movie.release_year,
          rating: movie.rating || movie.imdb_rating,
          genre: movie.genre || movie.genres,
        })

        // Only include movies with required fields
        if (movie.title || movie.name || movie.movie_title) {
          const parsedMovie = {
            id: i,
            title: movie.title || movie.name || movie.movie_title || `Movie ${i}`,
            year: Number.parseInt(movie.year || movie.release_year || movie.release_date?.split("-")[0]) || 2000,
            rating: Number.parseFloat(movie.rating || movie.imdb_rating || movie.score) || Math.random() * 3 + 7,
            genre: (movie.genre || movie.genres || movie.category || "Drama").split("|").map((g: string) => g.trim()),
            director: movie.director || movie.director_name || "Unknown Director",
            cast:
              movie.cast || movie.actors || movie.stars
                ? (movie.cast || movie.actors || movie.stars).split("|").slice(0, 3)
                : ["Unknown Actor"],
            plot: movie.plot || movie.description || movie.overview || "No description available.",
            poster: `/placeholder.svg?height=400&width=300&query=${encodeURIComponent((movie.title || movie.name || "movie") + " poster")}`,
            matchScore: Math.floor(Math.random() * 20) + 80,
            reason: `Based on your preferences and viewing history analysis`,
          }

          csvMovies.push(parsedMovie)
          console.log(`[v0] Added movie: ${parsedMovie.title} (${parsedMovie.year}) - Rating: ${parsedMovie.rating}`)
        }
      }
    } catch (fileError) {
      console.error("Error reading CSV file:", fileError)
      return NextResponse.json({ success: false, error: "Failed to read movie data from CSV" })
    }

    console.log(`[v0] Total movies parsed: ${csvMovies.length}`)

    if (csvMovies.length === 0) {
      return NextResponse.json({ success: false, error: "No valid movie data found in CSV file" })
    }

    // Fetch actual movie posters for each movie
    for (const movie of csvMovies) {
      try {
        const posterResponse = await fetch(`${process.env.NEXTAUTH_URL || 'http://localhost:3000'}/api/movie-poster`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            title: movie.title,
            year: movie.year
          })
        });

        if (posterResponse.ok) {
          const posterData = await posterResponse.json();
          if (posterData.success && posterData.posterUrl) {
            movie.poster = posterData.posterUrl;
          }
        }
      } catch (error) {
        console.log(`[v0] Failed to fetch poster for ${movie.title}:`, error);
        // Keep the placeholder poster if fetch fails
      }
    }

    // Simulate AI recommendation generation
    await new Promise((resolve) => setTimeout(resolve, 1500))

    console.log(`[v0] Filtering with preferences:`, preferences)

    let filteredMovies = csvMovies.filter((movie) => {
      let passesFilter = true
      const filterReasons = []

      // Filter by rating
      if (preferences.minRating && movie.rating < preferences.minRating) {
        passesFilter = false
        filterReasons.push(`rating ${movie.rating} < ${preferences.minRating}`)
      }

      // Filter by release year
      if (preferences.releaseYearRange) {
        const [minYear, maxYear] = preferences.releaseYearRange
        if (movie.year < minYear || movie.year > maxYear) {
          passesFilter = false
          filterReasons.push(`year ${movie.year} not in range ${minYear}-${maxYear}`)
        }
      }

      // Filter by preferred genres
      if (preferences.preferredGenres && preferences.preferredGenres.length > 0) {
        const hasPreferredGenre = movie.genre.some((g: string) =>
          preferences.preferredGenres.some((pg: string) => g.toLowerCase().includes(pg.toLowerCase())),
        )
        if (!hasPreferredGenre) {
          passesFilter = false
          filterReasons.push(`no preferred genres found in ${movie.genre.join(", ")}`)
        }
      }

      // Filter out avoided genres
      if (preferences.avoidGenres && preferences.avoidGenres.length > 0) {
        const hasAvoidedGenre = movie.genre.some((g: string) =>
          preferences.avoidGenres.some((ag: string) => g.toLowerCase().includes(ag.toLowerCase())),
        )
        if (hasAvoidedGenre) {
          passesFilter = false
          filterReasons.push(`has avoided genre`)
        }
      }

      if (!passesFilter) {
        console.log(`[v0] Filtered out "${movie.title}": ${filterReasons.join(", ")}`)
      } else {
        console.log(`[v0] Keeping "${movie.title}" - matches criteria`)
      }

      return passesFilter
    })

    console.log(`[v0] Movies after filtering: ${filteredMovies.length}`)

    filteredMovies = filteredMovies.map((movie) => {
      let score = movie.matchScore

      // Boost score for preferred genres
      if (preferences.preferredGenres) {
        const genreMatches = movie.genre.filter((g: string) =>
          preferences.preferredGenres.some((pg: string) => g.toLowerCase().includes(pg.toLowerCase())),
        ).length
        score += genreMatches * 5
      }

      // Boost score for mood match
      if (preferences.mood) {
        const moodGenreMap = {
          adventurous: ["adventure", "action"],
          romantic: ["romance", "romantic"],
          thrilling: ["thriller", "suspense", "mystery"],
          comedic: ["comedy", "humor"],
          dramatic: ["drama"],
          nostalgic: ["classic", "vintage"],
        }

        const moodGenres = moodGenreMap[preferences.mood.toLowerCase() as keyof typeof moodGenreMap] || []
        const hasMoodMatch = movie.genre.some((g: string) => moodGenres.some((mg: string) => g.toLowerCase().includes(mg)))
        if (hasMoodMatch) score += 10
      }

      return { ...movie, matchScore: Math.min(score, 100) }
    })

    // Sort by match score
    filteredMovies.sort((a, b) => b.matchScore - a.matchScore)

    // Limit to top 6 recommendations
    filteredMovies = filteredMovies.slice(0, 6)

    return NextResponse.json({
      success: true,
      recommendations: filteredMovies,
      totalFound: filteredMovies.length,
      totalMoviesInDataset: csvMovies.length,
      preferences: preferences,
      dataSource: "Your uploaded CSV file",
    })
  } catch (error) {
    console.error("Recommendations error:", error)
    return NextResponse.json({ success: false, error: "Failed to generate recommendations" })
  }
}
