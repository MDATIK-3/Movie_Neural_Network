import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { title, year } = await request.json()

    if (!title) {
      return NextResponse.json({
        success: true,
        posterUrl: `https://via.placeholder.com/300x450?text=Unknown+Movie`,
        title: "Unknown Movie",
      })
    }

    const tmdbApiKey = process.env.TMDB_API_KEY || "9ef5ae6fc8b8f484e9295dc97d8d32ea"

    if (tmdbApiKey === "demo_key") {
      return NextResponse.json({
        success: true,
        posterUrl: `https://via.placeholder.com/300x450?text=Demo+Poster`,
        title: title,
      })
    }

    console.log(`[v0] Fetching poster for: ${title} (${year})`)

    let searchUrl = `https://api.themoviedb.org/3/search/movie?api_key=${tmdbApiKey}&query=${encodeURIComponent(title)}`
    if (year) {
      searchUrl += `&year=${year}`
    }

    const searchResponse = await fetch(searchUrl, {
      method: "GET",
      headers: {
        Accept: "application/json",
        "User-Agent": "MovieApp/1.0",
      },
      signal: AbortSignal.timeout(5000), // Increased timeout for better reliability
    })

    if (!searchResponse.ok) {
      console.log(`[v0] TMDB API error for ${title}: ${searchResponse.status}`)

      if (searchResponse.status === 404) {
        return NextResponse.json({
          success: true,
          posterUrl: `https://via.placeholder.com/300x450?text=Movie+Not+Found`,
          title: title,
        })
      } else {
        return NextResponse.json({
          success: true,
          posterUrl: `https://via.placeholder.com/300x450?text=Error+Loading+Poster`,
          title: title,
        })
      }
    }

    const searchData = await searchResponse.json()
    console.log(`[v0] TMDB response for ${title}:`, searchData.results?.length || 0, "results")

    if (searchData.results && searchData.results.length > 0) {
      let selectedMovie = searchData.results[0]

      // If year is provided, try to find exact year match first
      if (year) {
        const yearMatch = searchData.results.find((movie: any) => {
          const movieYear = movie.release_date ? new Date(movie.release_date).getFullYear() : null
          return movieYear === Number.parseInt(year)
        })
        if (yearMatch) {
          selectedMovie = yearMatch
        }
      }

      if (selectedMovie.poster_path) {
        const posterUrl = `https://image.tmdb.org/t/p/w500${selectedMovie.poster_path}`
        console.log(`[v0] Found poster for ${title}: ${posterUrl}`)

        return NextResponse.json({
          success: true,
          posterUrl,
          tmdbId: selectedMovie.id,
          title: selectedMovie.title,
          overview: selectedMovie.overview,
          rating: selectedMovie.vote_average || 0,
        })
      }
    }

    console.log(`[v0] No poster found for ${title}, using placeholder`)
    return NextResponse.json({
      success: true,
      posterUrl: `https://via.placeholder.com/300x450?text=No+Poster`,
      title: title,
    })
  } catch (error) {
    console.log(
      `[v0] Poster fetch failed for movie, using placeholder:`,
      error instanceof Error ? error.message : "Unknown error",
    )

    const title = "Unknown Movie" // Declare the title variable here
    return NextResponse.json({
      success: true,
      posterUrl: `https://via.placeholder.com/300x450?text=Network+Error`,
      title: title || "Unknown Movie",
    })
  }
}
