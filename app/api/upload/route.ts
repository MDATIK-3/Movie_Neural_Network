import { type NextRequest, NextResponse } from "next/server"
import { writeFile } from "fs/promises"
import { join } from "path"

export async function POST(request: NextRequest) {
  try {
    const data = await request.formData()
    const file: File | null = data.get("file") as unknown as File

    if (!file) {
      return NextResponse.json({ success: false, error: "No file uploaded" })
    }

    // Validate file type
    if (!file.name.endsWith(".csv")) {
      return NextResponse.json({ success: false, error: "Please upload a CSV file" })
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      return NextResponse.json({ success: false, error: "File size must be less than 10MB" })
    }

    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)

    // Save file to uploads directory
    const uploadsDir = join(process.cwd(), "uploads")
    const filePath = join(uploadsDir, file.name)

    await writeFile(filePath, buffer)

    // Parse CSV to get basic info
    const csvContent = buffer.toString()
    const lines = csvContent.split("\n").filter((line) => line.trim())
    const headers = lines[0]?.split(",") || []
    const rowCount = lines.length - 1

    return NextResponse.json({
      success: true,
      fileName: file.name,
      fileSize: file.size,
      rowCount,
      columns: headers.length,
      headers: headers.slice(0, 10), // First 10 headers for preview
      filePath,
    })
  } catch (error) {
    console.error("Upload error:", error)
    return NextResponse.json({ success: false, error: "Failed to upload file" })
  }
}
