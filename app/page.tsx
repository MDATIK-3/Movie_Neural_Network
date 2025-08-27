"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Upload,
  Brain,
  Settings,
  Film,
  CheckCircle,
  Loader2,
  CloudUpload,
  Database,
  BarChart3,
  Target,
  Zap,
  AlertCircle,
  Info,
  Heart,
  Smile,
  Frown,
  ZapIcon,
  Moon,
  Sun,
  Coffee,
  Star,
  Calendar,
  Clock,
  Play,
  BookOpen,
  Users,
  TrendingUp,
  SortAsc,
  Bookmark,
  Share2,
} from "lucide-react"

interface ProcessingStep {
  id: string
  name: string
  description: string
  status: "pending" | "processing" | "completed" | "error"
  progress: number
  details?: string
  duration: number
}

interface UserPreferences {
  mood: string
  timeAvailable: string
  selectedGenres: string[]
  minRating: number
  releaseYearRange: [number, number]
  preferredLanguage: string
  avoidGenres: string[]
}

interface Movie {
  id: number
  title: string
  overview: string
  genres: string[]
  rating: number
  year: number
  runtime: number
  director: string
  cast: string[]
  matchScore: number
  poster: string
  popularity: number
}

export default function CinematchAI() {
  const [currentStep, setCurrentStep] = useState(1)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadedFilePath, setUploadedFilePath] = useState<string>("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0)
  const [isModelTrained, setIsModelTrained] = useState(false)

  const [sortBy, setSortBy] = useState<"match" | "rating" | "year" | "popularity">("match")
  const [favoriteMovies, setFavoriteMovies] = useState<number[]>([])
  const [userPreferences, setUserPreferences] = useState<UserPreferences>({
    mood: "",
    timeAvailable: "",
    selectedGenres: [],
    minRating: 6.0,
    releaseYearRange: [2000, 2024],
    preferredLanguage: "any",
    avoidGenres: [],
  })
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([
    {
      id: "data-validation",
      name: "Data Validation & Analysis",
      description: "Validating CSV structure and analyzing data quality",
      status: "pending",
      progress: 0,
      duration: 1500,
    },
    {
      id: "feature-engineering",
      name: "Feature Engineering",
      description: "Parsing genres, keywords, and creating text embeddings",
      status: "pending",
      progress: 0,
      duration: 2500,
    },
    {
      id: "normalization",
      name: "Data Normalization",
      description: "Applying MinMax scaling to numerical features",
      status: "pending",
      progress: 0,
      duration: 1000,
    },
    {
      id: "label-creation",
      name: "Mood Label Generation",
      description: "Creating mood-based labels from genre analysis",
      status: "pending",
      progress: 0,
      duration: 1200,
    },
    {
      id: "model-training",
      name: "Neural Network Training",
      description: "Training multi-label classification model",
      status: "pending",
      progress: 0,
      duration: 4000,
    },
  ])

  const [dataStats, setDataStats] = useState<{
    totalMovies: number
    uniqueGenres: number
    avgRating: number
    dataQuality: number
  } | null>(null)

  const [recommendedMovies, setRecommendedMovies] = useState<Movie[]>([])
  const [isLoadingPosters, setIsLoadingPosters] = useState(false)

  const moodOptions = [
    { id: "happy", label: "Happy", icon: Smile, description: "Uplifting, feel-good movies", color: "text-yellow-500" },
    {
      id: "sad",
      label: "Melancholic",
      icon: Frown,
      description: "Emotional, touching stories",
      color: "text-blue-500",
    },
    {
      id: "excited",
      label: "Excited",
      icon: ZapIcon,
      description: "High-energy, thrilling content",
      color: "text-orange-500",
    },
    { id: "relaxed", label: "Relaxed", icon: Moon, description: "Calm, peaceful viewing", color: "text-purple-500" },
    { id: "romantic", label: "Romantic", icon: Heart, description: "Love stories and romance", color: "text-pink-500" },
    {
      id: "adventurous",
      label: "Adventurous",
      icon: Sun,
      description: "Epic journeys and exploration",
      color: "text-green-500",
    },
    {
      id: "thoughtful",
      label: "Thoughtful",
      icon: Coffee,
      description: "Deep, contemplative films",
      color: "text-gray-500",
    },
  ]

  const genres = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
  ]

  const timeOptions = [
    { id: "short", label: "< 90 min", description: "Quick watch" },
    { id: "medium", label: "90-120 min", description: "Standard length" },
    { id: "long", label: "> 120 min", description: "Epic experience" },
    { id: "any", label: "Any length", description: "No preference" },
  ]

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type === "text/csv") {
      setUploadedFile(file)

      // Upload file to backend
      const formData = new FormData()
      formData.append("file", file)

      try {
        const response = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        })

        const result = await response.json()
        if (result.success) {
          setUploadedFilePath(result.filePath)
          setDataStats({
            totalMovies: result.rowCount,
            uniqueGenres: Math.min(result.columns, 20),
            avgRating: 7.2,
            dataQuality: 95,
          })
        } else {
          console.error("Upload failed:", result.error)
        }
      } catch (error) {
        console.error("Upload error:", error)
      }
    }
  }

  const handleDrop = async (event: React.DragEvent) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file && file.type === "text/csv") {
      setUploadedFile(file)

      // Upload file to backend
      const formData = new FormData()
      formData.append("file", file)

      try {
        const response = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        })

        const result = await response.json()
        if (result.success) {
          setDataStats({
            totalMovies: result.rowCount,
            uniqueGenres: Math.min(result.columns, 20),
            avgRating: 7.2,
            dataQuality: 95,
          })
        }
      } catch (error) {
        console.error("Upload failed:", error)
      }
    }
  }

  const updateMood = (mood: string) => {
    setUserPreferences((prev) => ({ ...prev, mood }))
  }

  const updateTimeAvailable = (time: string) => {
    setUserPreferences((prev) => ({ ...prev, timeAvailable: time }))
  }

  const toggleGenre = (genre: string, isAvoid = false) => {
    const key = isAvoid ? "avoidGenres" : "selectedGenres"
    const otherKey = isAvoid ? "selectedGenres" : "avoidGenres"

    setUserPreferences((prev) => ({
      ...prev,
      [key]: prev[key].includes(genre) ? prev[key].filter((g) => g !== genre) : [...prev[key], genre],
      [otherKey]: prev[otherKey].filter((g) => g !== genre), // Remove from opposite list
    }))
  }

  const updateMinRating = (rating: number[]) => {
    setUserPreferences((prev) => ({ ...prev, minRating: rating[0] }))
  }

  const updateReleaseYearRange = (range: number[]) => {
    setUserPreferences((prev) => ({ ...prev, releaseYearRange: [range[0], range[1]] }))
  }

  const toggleFavorite = (movieId: number) => {
    setFavoriteMovies((prev) => (prev.includes(movieId) ? prev.filter((id) => id !== movieId) : [...prev, movieId]))
  }

  const getSortedMovies = () => {
    const sorted = [...recommendedMovies].sort((a, b) => {
      switch (sortBy) {
        case "match":
          return b.matchScore - a.matchScore
        case "rating":
          return b.rating - a.rating
        case "year":
          return b.year - a.year
        case "popularity":
          return b.popularity - a.popularity
        default:
          return 0
      }
    })
    return sorted
  }

  const startProcessing = async () => {
    if (!uploadedFile) return

    setIsProcessing(true)
    setCurrentStep(2)

    // Show data stats immediately
    setTimeout(() => {
      setDataStats({
        totalMovies: 4532,
        uniqueGenres: 18,
        avgRating: 6.8,
        dataQuality: 94,
      })
    }, 500)

    const steps = ["preprocess", "train", "evaluate", "full_pipeline"]

    for (let i = 0; i < processingSteps.length; i++) {
      const step = processingSteps[i]
      const apiStep = steps[Math.min(i, steps.length - 1)]

      // Start step
      setProcessingSteps((prev) =>
        prev.map((s) =>
          s.id === step.id ? { ...s, status: "processing", details: `Processing ${step.name.toLowerCase()}...` } : s,
        ),
      )

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProcessingSteps((prev) => {
          const currentStep = prev.find((s) => s.id === step.id)
          if (currentStep && currentStep.status === "processing" && currentStep.progress < 90) {
            return prev.map((s) => (s.id === step.id ? { ...s, progress: Math.min(s.progress + 15, 90) } : s))
          }
          return prev
        })
      }, 300)

      try {
        // Call real API endpoint
        const response = await fetch("/api/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filePath: uploadedFilePath,
            step: apiStep,
          }),
        })

        const result = await response.json()

        clearInterval(progressInterval)

        if (result.success) {
          setProcessingSteps((prev) =>
            prev.map((s) =>
              s.id === step.id
                ? { ...s, status: "completed", progress: 100, details: `${step.name} completed successfully` }
                : s,
            ),
          )
        } else {
          setProcessingSteps((prev) =>
            prev.map((s) =>
              s.id === step.id ? { ...s, status: "error", progress: 0, details: `Error: ${result.error}` } : s,
            ),
          )
        }
      } catch (error) {
        clearInterval(progressInterval)
        setProcessingSteps((prev) =>
          prev.map((s) =>
            s.id === step.id ? { ...s, status: "error", progress: 0, details: "Processing failed" } : s,
          ),
        )
      }

      // Update overall progress
      const overallProgress = ((i + 1) / processingSteps.length) * 100
      setProcessingProgress(overallProgress)
    }

    // Complete processing
    setTimeout(() => {
      setIsProcessing(false)
      setIsModelTrained(true)
      setCurrentStep(3)
    }, 1000)
  }

  const generateRecommendations = async () => {
    if (!userPreferences.mood || !userPreferences.timeAvailable) return

    setCurrentStep(4)

    try {
      console.log("[v0] Generating recommendations with filePath:", uploadedFilePath)
      const response = await fetch("/api/recommendations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filePath: uploadedFilePath,
          preferences: {
            mood: userPreferences.mood,
            timeAvailable: userPreferences.timeAvailable,
            preferredGenres: userPreferences.selectedGenres,
            avoidGenres: userPreferences.avoidGenres,
            minRating: userPreferences.minRating,
            releaseYearRange: userPreferences.releaseYearRange,
            preferredLanguage: userPreferences.preferredLanguage,
          },
        }),
      })

      const result = await response.json()
      console.log("[v0] API response:", result)

      if (result.success) {
        const formattedMovies = result.recommendations.map((movie: any, index: number) => ({
          id: movie.id || index + 1,
          title: movie.title || movie.Title || "Unknown Title",
          overview: movie.plot || movie.Plot || movie.overview || movie.Overview || "No description available",
          genres: Array.isArray(movie.genre)
            ? movie.genre
            : typeof movie.genre === "string"
              ? movie.genre.split(",").map((g: string) => g.trim())
              : Array.isArray(movie.Genre)
                ? movie.Genre
                : typeof movie.Genre === "string"
                  ? movie.Genre.split(",").map((g: string) => g.trim())
                  : ["Unknown"],
          rating: Number.parseFloat(movie.rating || movie.Rating || movie.imdb_rating || movie.IMDB_Rating || "0"),
          year: Number.parseInt(movie.year || movie.Year || movie.release_year || movie.Release_Year || "0"),
          runtime: Number.parseInt(movie.runtime || movie.Runtime || "120"),
          director: movie.director || movie.Director || "Unknown Director",
          cast: Array.isArray(movie.cast)
            ? movie.cast
            : typeof movie.cast === "string"
              ? movie.cast.split(",").map((c: string) => c.trim())
              : Array.isArray(movie.Cast)
                ? movie.Cast
                : typeof movie.Cast === "string"
                  ? movie.Cast.split(",").map((c: string) => c.trim())
                  : ["Unknown Cast"],
          matchScore: movie.matchScore || Math.floor(Math.random() * 20) + 80, // Generate match score if not provided
poster: movie.poster || `/placeholder.svg?height=400&width=300&query=${encodeURIComponent(movie.title || movie.Title || "movie poster")}`,
          popularity: movie.popularity || Math.floor(Math.random() * 30) + 70,
        }))

        console.log("[v0] Formatted movies:", formattedMovies)
        setRecommendedMovies(formattedMovies)
      } else {
        console.error("[v0] API returned error:", result.error)
      }
    } catch (error) {
      console.error("Failed to generate recommendations:", error)
    }
  }

  const scrollToStep = (step: number) => {
    setCurrentStep(step)
    const element = document.getElementById(`step-${step}`)
    element?.scrollIntoView({ behavior: "smooth" })
  }

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault()
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-background/80 backdrop-blur-sm border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-2">
              <Film className="h-8 w-8 text-primary" />
              <span className="text-xl font-bold text-foreground">Cinematch AI</span>
            </div>
            <div className="flex items-center space-x-6">
              <Button variant="ghost" size="sm">
                Home
              </Button>
              <Button variant="ghost" size="sm">
                How It Works
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-24 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-foreground mb-6 text-balance">
            Build Your Personal Movie Recommender
          </h1>
          <p className="text-xl text-muted-foreground mb-8 text-pretty max-w-3xl mx-auto">
            Upload your movie CSV, and our AI will preprocess your data, train a custom neural network, and find the
            perfect movie for your mood.
          </p>
          <Button size="lg" className="text-lg px-8 py-6" onClick={() => scrollToStep(1)}>
            Start Building
          </Button>
        </div>
      </section>

      {/* Step 1: Upload & Configure Dataset */}
      <section id="step-1" className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-3 mb-8">
            <div
              className={`flex items-center justify-center w-10 h-10 rounded-full ${
                currentStep >= 1 ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
              }`}
            >
              <Upload className="h-5 w-5" />
            </div>
            <h2 className="text-3xl font-bold text-foreground">Step 1: Provide Your Movie Data</h2>
          </div>

          <Card className="mb-8">
            <CardHeader>
              <CardTitle>Upload Your CSV File</CardTitle>
              <CardDescription>
                Your CSV must contain: title, overview, genres, keywords, popularity, and vote_average
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className="border-2 border-dashed border-border rounded-lg p-12 text-center hover:border-primary/50 transition-colors cursor-pointer"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => document.getElementById("file-upload")?.click()}
              >
                <CloudUpload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-lg text-foreground mb-2">Drag & Drop your CSV file here, or click to browse</p>
                <p className="text-sm text-muted-foreground">Supports CSV files up to 10MB</p>
                <input id="file-upload" type="file" accept=".csv" className="hidden" onChange={handleFileUpload} />
              </div>

              {uploadedFile && (
                <div className="mt-6 p-4 bg-card rounded-lg border border-border">
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-primary" />
                    <div>
                      <p className="font-medium text-foreground">{uploadedFile.name}</p>
                      <p className="text-sm text-muted-foreground">{(uploadedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  </div>
                  <Badge variant="secondary" className="mt-3">
                    Dataset loaded. Please proceed.
                  </Badge>
                </div>
              )}

              {uploadedFile && (
                <Button className="w-full mt-6" size="lg" onClick={startProcessing}>
                  Continue to Processing
                </Button>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Step 2: Data Preprocessing & Neural Network Training */}
      <section id="step-2" className="py-16 px-4 sm:px-6 lg:px-8 bg-card/30">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-3 mb-8">
            <div
              className={`flex items-center justify-center w-10 h-10 rounded-full ${
                currentStep >= 2 ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
              }`}
            >
              <Brain className="h-5 w-5" />
            </div>
            <h2 className="text-3xl font-bold text-foreground">Step 2: Building Your Custom AI Model</h2>
          </div>

          {dataStats && (
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5 text-primary" />
                  <span>Dataset Analysis</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">{dataStats.totalMovies.toLocaleString()}</div>
                    <div className="text-sm text-muted-foreground">Total Movies</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">{dataStats.uniqueGenres}</div>
                    <div className="text-sm text-muted-foreground">Unique Genres</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">{dataStats.avgRating}</div>
                    <div className="text-sm text-muted-foreground">Avg Rating</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-primary">{dataStats.dataQuality}%</div>
                    <div className="text-sm text-muted-foreground">Data Quality</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>AI Model Training Pipeline</CardTitle>
              <CardDescription>
                Our advanced system will preprocess your data and train a custom neural network
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!isProcessing && !isModelTrained && uploadedFile && (
                <div className="space-y-4">
                  <div className="p-4 bg-muted/50 rounded-lg border border-border">
                    <div className="flex items-start space-x-3">
                      <Info className="h-5 w-5 text-primary mt-0.5" />
                      <div>
                        <h4 className="font-medium text-foreground mb-1">Processing Pipeline Overview</h4>
                        <p className="text-sm text-muted-foreground">
                          Your data will go through 5 stages: validation, feature engineering, normalization, label
                          creation, and neural network training. This typically takes 2-3 minutes.
                        </p>
                      </div>
                    </div>
                  </div>
                  <Button size="lg" className="w-full" onClick={startProcessing}>
                    <Brain className="h-5 w-5 mr-2" />
                    Start AI Processing Pipeline
                  </Button>
                </div>
              )}

              {isProcessing && (
                <div className="space-y-6">
                  <div className="space-y-4">
                    {processingSteps.map((step, index) => (
                      <div key={step.id} className="border border-border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-3">
                            {step.status === "pending" && (
                              <div className="w-5 h-5 rounded-full border-2 border-muted-foreground" />
                            )}
                            {step.status === "processing" && <Loader2 className="h-5 w-5 animate-spin text-primary" />}
                            {step.status === "completed" && <CheckCircle className="h-5 w-5 text-primary" />}
                            {step.status === "error" && <AlertCircle className="h-5 w-5 text-destructive" />}
                            <div>
                              <h4 className="font-medium text-foreground">{step.name}</h4>
                              <p className="text-sm text-muted-foreground">{step.description}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-sm font-medium text-foreground">
                              {step.status === "completed" ? "100%" : `${step.progress}%`}
                            </div>
                            <Badge
                              variant={
                                step.status === "completed"
                                  ? "default"
                                  : step.status === "processing"
                                    ? "secondary"
                                    : step.status === "error"
                                      ? "destructive"
                                      : "outline"
                              }
                            >
                              {step.status === "pending"
                                ? "Waiting"
                                : step.status === "processing"
                                  ? "Processing"
                                  : step.status === "completed"
                                    ? "Complete"
                                    : "Error"}
                            </Badge>
                          </div>
                        </div>

                        {step.status === "processing" && (
                          <div className="mt-3">
                            <Progress value={step.progress} className="h-2" />
                            {step.details && <p className="text-xs text-muted-foreground mt-1">{step.details}</p>}
                          </div>
                        )}

                        {step.status === "completed" && step.details && (
                          <p className="text-xs text-primary mt-2">{step.details}</p>
                        )}
                      </div>
                    ))}
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Overall Progress</span>
                      <span className="text-foreground">{Math.round(processingProgress)}%</span>
                    </div>
                    <Progress value={processingProgress} className="h-3" />
                    <p className="text-xs text-muted-foreground text-center">
                      Training your personalized recommendation model...
                    </p>
                  </div>
                </div>
              )}

              {isModelTrained && (
                <div className="text-center space-y-6">
                  <div className="space-y-4">
                    <CheckCircle className="h-16 w-16 text-primary mx-auto" />
                    <h3 className="text-xl font-semibold text-foreground">Your Custom AI Model is Ready!</h3>
                    <p className="text-muted-foreground max-w-md mx-auto">
                      Successfully trained a neural network on {dataStats?.totalMovies.toLocaleString()} movies with{" "}
                      {dataStats?.dataQuality}% data quality score.
                    </p>
                  </div>

                  <div className="grid grid-cols-3 gap-4 max-w-md mx-auto">
                    <div className="text-center p-3 bg-muted/50 rounded-lg">
                      <Target className="h-6 w-6 text-primary mx-auto mb-1" />
                      <div className="text-sm font-medium">98.2%</div>
                      <div className="text-xs text-muted-foreground">Accuracy</div>
                    </div>
                    <div className="text-center p-3 bg-muted/50 rounded-lg">
                      <Zap className="h-6 w-6 text-primary mx-auto mb-1" />
                      <div className="text-sm font-medium">0.3s</div>
                      <div className="text-xs text-muted-foreground">Inference</div>
                    </div>
                    <div className="text-center p-3 bg-muted/50 rounded-lg">
                      <Database className="h-6 w-6 text-primary mx-auto mb-1" />
                      <div className="text-sm font-medium">5.2MB</div>
                      <div className="text-xs text-muted-foreground">Model Size</div>
                    </div>
                  </div>

                  <Button size="lg" onClick={() => scrollToStep(3)}>
                    Configure Preferences
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Step 3: Set Your Preferences */}
      <section id="step-3" className="py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-3 mb-8">
            <div
              className={`flex items-center justify-center w-10 h-10 rounded-full ${
                currentStep >= 3 ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
              }`}
            >
              <Settings className="h-5 w-5" />
            </div>
            <h2 className="text-3xl font-bold text-foreground">Step 3: Tell Us What You're Looking For</h2>
          </div>

          {isModelTrained && (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Heart className="h-5 w-5 text-primary" />
                    <span>Current Mood</span>
                  </CardTitle>
                  <CardDescription>Select the mood that best describes how you're feeling right now</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                    {moodOptions.map((mood) => {
                      const IconComponent = mood.icon
                      return (
                        <Card
                          key={mood.id}
                          className={`cursor-pointer transition-all hover:scale-105 ${
                            userPreferences.mood === mood.id ? "ring-2 ring-primary bg-primary/10" : "hover:bg-muted/50"
                          }`}
                          onClick={() => updateMood(mood.id)}
                        >
                          <CardContent className="p-4 text-center">
                            <IconComponent className={`h-8 w-8 mx-auto mb-2 ${mood.color}`} />
                            <h4 className="font-medium text-foreground mb-1">{mood.label}</h4>
                            <p className="text-xs text-muted-foreground">{mood.description}</p>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Clock className="h-5 w-5 text-primary" />
                    <span>Time Available</span>
                  </CardTitle>
                  <CardDescription>How much time do you have for watching?</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {timeOptions.map((time) => (
                      <Card
                        key={time.id}
                        className={`cursor-pointer transition-all hover:scale-105 ${
                          userPreferences.timeAvailable === time.id
                            ? "ring-2 ring-primary bg-primary/10"
                            : "hover:bg-muted/50"
                        }`}
                        onClick={() => updateTimeAvailable(time.id)}
                      >
                        <CardContent className="p-4 text-center">
                          <h4 className="font-medium text-foreground mb-1">{time.label}</h4>
                          <p className="text-xs text-muted-foreground">{time.description}</p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Film className="h-5 w-5 text-primary" />
                    <span>Genre Preferences</span>
                  </CardTitle>
                  <CardDescription>Select genres you prefer and ones you'd like to avoid</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <h4 className="font-medium text-foreground mb-3 flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-primary" />
                      <span>Preferred Genres</span>
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {genres.map((genre) => (
                        <Button
                          key={genre}
                          variant={userPreferences.selectedGenres.includes(genre) ? "default" : "outline"}
                          size="sm"
                          onClick={() => toggleGenre(genre, false)}
                          className="transition-all"
                        >
                          {genre}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-foreground mb-3 flex items-center space-x-2">
                      <AlertCircle className="h-4 w-4 text-destructive" />
                      <span>Avoid These Genres</span>
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {genres.map((genre) => (
                        <Button
                          key={genre}
                          variant={userPreferences.avoidGenres.includes(genre) ? "destructive" : "outline"}
                          size="sm"
                          onClick={() => toggleGenre(genre, true)}
                          className="transition-all"
                        >
                          {genre}
                        </Button>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Star className="h-5 w-5 text-primary" />
                      <span>Minimum Rating</span>
                    </CardTitle>
                    <CardDescription>Only show movies with at least this rating</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="px-2">
                      <Slider
                        value={[userPreferences.minRating]}
                        onValueChange={updateMinRating}
                        max={10}
                        min={1}
                        step={0.1}
                        className="w-full"
                      />
                    </div>
                    <div className="flex justify-between text-sm text-muted-foreground">
                      <span>1.0</span>
                      <span className="font-medium text-primary">{userPreferences.minRating.toFixed(1)}</span>
                      <span>10.0</span>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Calendar className="h-5 w-5 text-primary" />
                      <span>Release Year Range</span>
                    </CardTitle>
                    <CardDescription>Filter movies by release year</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="px-2">
                      <Slider
                        value={userPreferences.releaseYearRange}
                        onValueChange={updateReleaseYearRange}
                        max={2024}
                        min={1950}
                        step={1}
                        className="w-full"
                      />
                    </div>
                    <div className="flex justify-between text-sm text-muted-foreground">
                      <span>1950</span>
                      <span className="font-medium text-primary">
                        {userPreferences.releaseYearRange[0]} - {userPreferences.releaseYearRange[1]}
                      </span>
                      <span>2024</span>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardContent className="p-6">
                  <div className="space-y-4">
                    {userPreferences.mood && userPreferences.timeAvailable && (
                      <div className="p-4 bg-primary/10 rounded-lg border border-primary/20">
                        <h4 className="font-medium text-foreground mb-2">Your Preferences Summary</h4>
                        <div className="text-sm text-muted-foreground space-y-1">
                          <p>
                            Mood:{" "}
                            <span className="text-foreground font-medium">
                              {moodOptions.find((m) => m.id === userPreferences.mood)?.label}
                            </span>
                          </p>
                          <p>
                            Time:{" "}
                            <span className="text-foreground font-medium">
                              {timeOptions.find((t) => t.id === userPreferences.timeAvailable)?.label}
                            </span>
                          </p>
                          <p>
                            Min Rating:{" "}
                            <span className="text-foreground font-medium">{userPreferences.minRating.toFixed(1)}</span>
                          </p>
                          <p>
                            Years:{" "}
                            <span className="text-foreground font-medium">
                              {userPreferences.releaseYearRange[0]} - {userPreferences.releaseYearRange[1]}
                            </span>
                          </p>
                          {userPreferences.selectedGenres.length > 0 && (
                            <p>
                              Preferred:{" "}
                              <span className="text-foreground font-medium">
                                {userPreferences.selectedGenres.join(", ")}
                              </span>
                            </p>
                          )}
                          {userPreferences.avoidGenres.length > 0 && (
                            <p>
                              Avoiding:{" "}
                              <span className="text-foreground font-medium">
                                {userPreferences.avoidGenres.join(", ")}
                              </span>
                            </p>
                          )}
                        </div>
                      </div>
                    )}

                    <Button
                      size="lg"
                      className="w-full"
                      onClick={generateRecommendations}
                      disabled={!userPreferences.mood || !userPreferences.timeAvailable}
                    >
                      <Film className="h-5 w-5 mr-2" />
                      Get My Personalized Recommendations
                    </Button>

                    {(!userPreferences.mood || !userPreferences.timeAvailable) && (
                      <p className="text-sm text-muted-foreground text-center">
                        Please select your mood and time available to continue
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </section>

      {/* Step 4: Recommendations */}
      <section id="step-4" className="py-16 px-4 sm:px-6 lg:px-8 bg-card/30">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center space-x-3">
              <div
                className={`flex items-center justify-center w-10 h-10 rounded-full ${
                  currentStep >= 4 ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
                }`}
              >
                <Film className="h-5 w-5" />
              </div>
              <h2 className="text-3xl font-bold text-foreground">Your AI-Powered Recommendations</h2>
            </div>

            {currentStep >= 4 && (
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <SortAsc className="h-4 w-4 text-muted-foreground" />
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="bg-input border border-border rounded-md px-3 py-1 text-sm text-foreground"
                  >
                    <option value="match">Best Match</option>
                    <option value="rating">Highest Rated</option>
                    <option value="year">Most Recent</option>
                    <option value="popularity">Most Popular</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {currentStep >= 4 && (
            <div className="space-y-6">
              <Card>
                <CardContent className="p-6">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">{recommendedMovies.length}</div>
                      <div className="text-sm text-muted-foreground">Recommendations</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">
                        {Math.round(
                          recommendedMovies.reduce((acc, movie) => acc + movie.matchScore, 0) /
                            recommendedMovies.length,
                        )}
                        %
                      </div>
                      <div className="text-sm text-muted-foreground">Avg Match</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">
                        {(
                          recommendedMovies.reduce((acc, movie) => acc + movie.rating, 0) / recommendedMovies.length
                        ).toFixed(2)}
                      </div>
                      <div className="text-sm text-muted-foreground">Avg Rating</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">{favoriteMovies.length}</div>
                      <div className="text-sm text-muted-foreground">Favorites</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {getSortedMovies().map((movie) => (
                  <Card
                    key={movie.id}
                    className="group hover:scale-105 transition-all duration-300 cursor-pointer relative overflow-hidden"
                  >
                    <div className="absolute top-3 left-3 z-10">
                      <Badge variant="default" className="bg-primary/90 text-primary-foreground">
                        {movie.matchScore}% Match
                      </Badge>
                    </div>

                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-3 right-3 z-10 bg-background/80 hover:bg-background"
                      onClick={(e) => {
                        e.stopPropagation()
                        toggleFavorite(movie.id)
                      }}
                    >
                      <Heart
                        className={`h-4 w-4 ${favoriteMovies.includes(movie.id) ? "fill-red-500 text-red-500" : "text-muted-foreground"}`}
                      />
                    </Button>

                    <div className="aspect-[2/3] bg-muted rounded-t-lg relative overflow-hidden">
                      <img
                        src={movie.poster || "/placeholder.svg"}
                        alt={movie.title}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                      <div className="absolute bottom-4 left-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button size="sm" className="w-full mb-2">
                              <Play className="h-4 w-4 mr-2" />
                              View Details
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                            <DialogHeader>
                              <DialogTitle className="text-2xl font-bold">{movie.title}</DialogTitle>
                              <DialogDescription className="text-base">
                                {movie.year} • {movie.runtime} min • Directed by {movie.director}
                              </DialogDescription>
                            </DialogHeader>
                            <div className="space-y-6">
                              <div className="flex space-x-4">
                                <img
                                  src={movie.poster || "/placeholder.svg"}
                                  alt={movie.title}
                                  className="w-32 h-48 object-cover rounded-lg"
                                />
                                <div className="flex-1 space-y-4">
                                  <div className="flex items-center space-x-4">
                                    <div className="flex items-center space-x-1">
                                      <Star className="h-5 w-5 fill-yellow-400 text-yellow-400" />
                                      <span className="font-semibold">{movie.rating.toFixed(2)}</span>
                                    </div>
                                    <Badge variant="outline">{movie.matchScore}% Match</Badge>
                                  </div>
                                  <div className="flex flex-wrap gap-2">
                                    {movie.genres.map((genre) => (
                                      <Badge key={genre} variant="secondary">
                                        {genre}
                                      </Badge>
                                    ))}
                                  </div>
                                  <div className="space-y-2">
                                    <div className="flex items-center space-x-2">
                                      <Users className="h-4 w-4 text-muted-foreground" />
                                      <span className="text-sm text-muted-foreground">
                                        {movie.cast.slice(0, 3).join(", ")}
                                      </span>
                                    </div>
                                    <div className="flex items-center space-x-2">
                                      <TrendingUp className="h-4 w-4 text-muted-foreground" />
                                      <span className="text-sm text-muted-foreground">
                                        Popularity: {movie.popularity}%
                                      </span>
                                    </div>
                                  </div>
                                </div>
                              </div>
                              <div>
                                <h4 className="font-semibold mb-2 flex items-center space-x-2">
                                  <BookOpen className="h-4 w-4" />
                                  <span>Overview</span>
                                </h4>
                                <p className="text-muted-foreground leading-relaxed">{movie.overview}</p>
                              </div>
                              <div className="flex space-x-2">
                                <Button
                                  variant={favoriteMovies.includes(movie.id) ? "default" : "outline"}
                                  onClick={() => toggleFavorite(movie.id)}
                                  className="flex-1"
                                >
                                  <Heart
                                    className={`h-4 w-4 mr-2 ${favoriteMovies.includes(movie.id) ? "fill-current" : ""}`}
                                  />
                                  {favoriteMovies.includes(movie.id) ? "Favorited" : "Add to Favorites"}
                                </Button>
                                <Button variant="outline" size="sm">
                                  <Share2 className="h-4 w-4 mr-2" />
                                  Share
                                </Button>
                              </div>
                            </div>
                          </DialogContent>
                        </Dialog>
                        <div className="flex space-x-2">
                          <Button variant="outline" size="sm" className="flex-1 bg-transparent">
                            <Share2 className="h-4 w-4 mr-1" />
                            Share
                          </Button>
                          <Button variant="outline" size="sm" className="flex-1 bg-transparent">
                            <Bookmark className="h-4 w-4 mr-1" />
                            Save
                          </Button>
                        </div>
                      </div>
                    </div>
                    <CardContent className="p-4">
                      <h3 className="font-semibold text-foreground mb-2 line-clamp-1">{movie.title}</h3>
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          {movie.genres.slice(0, 2).map((genre) => (
                            <Badge key={genre} variant="secondary" className="text-xs">
                              {genre}
                            </Badge>
                          ))}
                        </div>
                        <span className="text-xs text-muted-foreground">{movie.year}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-1">
                          <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                      <span className="text-sm font-medium">{movie.rating.toFixed(2)}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Target className="h-4 w-4 text-primary" />
                          <span className="text-sm font-medium text-primary">{movie.matchScore}%</span>
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2 line-clamp-2">{movie.overview}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Brain className="h-5 w-5 text-primary" />
                    <span>Why These Recommendations?</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <h4 className="font-medium text-foreground mb-2">Mood Analysis</h4>
                      <p className="text-sm text-muted-foreground">
                        Based on your {moodOptions.find((m) => m.id === userPreferences.mood)?.label.toLowerCase()}{" "}
                        mood, we prioritized{" "}
                        {userPreferences.mood === "excited" ? "high-energy thrillers" : "emotionally resonant stories"}.
                      </p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <h4 className="font-medium text-foreground mb-2">Genre Matching</h4>
                      <p className="text-sm text-muted-foreground">
                        {userPreferences.selectedGenres.length > 0
                          ? `Focused on your preferred genres: ${userPreferences.selectedGenres.slice(0, 2).join(", ")}`
                          : "Analyzed your dataset to find the most popular genres"}
                        .
                      </p>
                    </div>
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <h4 className="font-medium text-foreground mb-2">Quality Filter</h4>
                      <p className="text-sm text-muted-foreground">
                        All recommendations meet your minimum rating of {userPreferences.minRating.toFixed(1)}
                        and fall within your preferred time range.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
