package main

import (
	"encoding/json"
	"image/color"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

// Button represents a clickable UI element with a label and an associated action.
type Button struct {
	x, y, width, height int
	label               string
	action              func()
}

// Game holds the state of the simulation, including the grid, rules, and UI elements.
type Game struct {
	grid         [][]uint8
	tempGrid     [][]uint8
	rules        map[string]bool
	buttons      []Button
	cellSize     int
	windowWidth  int
	windowHeight int
}

// WindowState represents the saved state of the application window, including its size and position.
type WindowState struct {
	Width      int  `json:"width"`
	Height     int  `json:"height"`
	X          int  `json:"x"`
	Y          int  `json:"y"`
	Fullscreen bool `json:"fullscreen"`
}

// NewGame initializes a new Game instance with the specified window dimensions and cell size.
func NewGame(windowWidth, windowHeight, cellSize int) *Game {
	game := &Game{
		cellSize:     cellSize,
		windowWidth:  windowWidth,
		windowHeight: windowHeight,
		rules: map[string]bool{
			"game_of_life": true,
			"trees":        false,
		},
	}
	game.resizeGrid()
	game.setupUI()
	return game
}

// resizeGrid dynamically adjusts the simulation grid dimensions based on the current window size.
// It calculates the new grid width and height by dividing the window dimensions by the cell size,
// then reallocates both the main grid and temporary grid arrays to match these new dimensions.
// This function is typically called when the window is resized to maintain proper grid scaling.
func (g *Game) resizeGrid() {
	gridHeight := g.windowHeight / g.cellSize
	gridWidth := g.windowWidth / g.cellSize
	g.grid = make([][]uint8, gridHeight)
	g.tempGrid = make([][]uint8, gridHeight)
	for i := range g.grid {
		g.grid[i] = make([]uint8, gridWidth)
		g.tempGrid[i] = make([]uint8, gridWidth)
	}
}

// setupUI initializes the buttons for the game UI.
func (g *Game) setupUI() {
	g.buttons = []Button{
		{
			x: 20, y: 20, width: 200, height: 50, label: "Toggle Game of Life",
			action: func() { g.rules["game_of_life"] = !g.rules["game_of_life"] },
		},
		{
			x: 20, y: 80, width: 200, height: 50, label: "Toggle Trees",
			action: func() { g.rules["trees"] = !g.rules["trees"] },
		},
		{
			x: 20, y: 140, width: 200, height: 50, label: "Randomize Grid",
			action: func() { g.randomizeGrid(0.1) },
		},
	}
}

// randomizeGrid populates the grid with random values based on the specified density.
func (g *Game) randomizeGrid(density float64) {
	for y := 0; y < len(g.grid); y++ {
		for x := 0; x < len(g.grid[0]); x++ {
			if rand.Float64() < density {
				g.grid[y][x] = 1
			} else {
				g.grid[y][x] = 0
			}
		}
	}
}

// applyGameOfLifeRule updates the grid based on the Game of Life rules.
func (g *Game) applyGameOfLifeRule() {
	if !g.rules["game_of_life"] {
		return
	}

	for y := 0; y < len(g.grid); y++ {
		for x := 0; x < len(g.grid[0]); x++ {
			neighbors := g.countNeighbors(x, y, 1)
			if g.grid[y][x] == 1 {
				if neighbors < 2 || neighbors > 3 {
					g.tempGrid[y][x] = 0
				} else {
					g.tempGrid[y][x] = 1
				}
			} else {
				if neighbors == 3 {
					g.tempGrid[y][x] = 1
				} else {
					g.tempGrid[y][x] = 0
				}
			}
		}
	}

	g.swapGrids()
}

// applyTreesRule updates the grid based on the Trees simulation rules.
func (g *Game) applyTreesRule() {
	if !g.rules["trees"] {
		return
	}

	rand.Seed(time.Now().UnixNano())
	for y := 0; y < len(g.grid); y++ {
		for x := 0; x < len(g.grid[0]); x++ {
			if g.grid[y][x] == 0 && rand.Float64() < 0.002 {
				g.tempGrid[y][x] = 2
			} else if g.grid[y][x] == 2 {
				g.tempGrid[y][x] = 2
			}
		}
	}

	g.swapGrids()
}

// countNeighbors counts the number of neighboring cells in the specified state.
func (g *Game) countNeighbors(x, y, state int) int {
	count := 0
	dirs := [][2]int{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1}, {0, 1},
		{1, -1}, {1, 0}, {1, 1},
	}

	for _, dir := range dirs {
		nx, ny := x+dir[0], y+dir[1]
		if nx >= 0 && nx < len(g.grid[0]) && ny >= 0 && ny < len(g.grid) {
			if int(g.grid[ny][nx]) == state {
				count++
			}
		}
	}

	return count
}

// swapGrids swaps the main grid with the temporary grid.
func (g *Game) swapGrids() {
	g.grid, g.tempGrid = g.tempGrid, g.grid
}

// Update processes user input and applies simulation rules.
func (g *Game) Update() error {
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		mx, my := ebiten.CursorPosition()
		for _, button := range g.buttons {
			if mx >= button.x && mx <= button.x+button.width && my >= button.y && my <= button.y+button.height {
				button.action()
			}
		}
	}

	g.applyTreesRule()
	g.applyGameOfLifeRule()
	return nil
}

// Draw renders the grid and UI elements on the screen.
func (g *Game) Draw(screen *ebiten.Image) {
	green := color.RGBA{0, 255, 0, 255} // Define green color
	screen.Fill(color.Black)
	for y := 0; y < len(g.grid); y++ {
		for x := 0; x < len(g.grid[0]); x++ {
			switch g.grid[y][x] {
			case 1:
				ebitenutil.DrawRect(screen, float64(x*g.cellSize), float64(y*g.cellSize), float64(g.cellSize), float64(g.cellSize), color.White)
			case 2:
				ebitenutil.DrawRect(screen, float64(x*g.cellSize), float64(y*g.cellSize), float64(g.cellSize), float64(g.cellSize), green)
			}
		}
	}

	// Draw buttons
	for _, button := range g.buttons {
		btnColor := color.RGBA{128, 128, 128, 255}
		ebitenutil.DrawRect(screen, float64(button.x), float64(button.y), float64(button.width), float64(button.height), btnColor)
		ebitenutil.DebugPrintAt(screen, button.label, button.x+10, button.y+10)
	}
}

// Layout adjusts the game layout based on the window size.
func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	if outsideWidth != g.windowWidth || outsideHeight != g.windowHeight {
		// Update the window dimensions
		g.windowWidth = outsideWidth
		g.windowHeight = outsideHeight
		g.resizeGrid()

		// Save the window state on resize
		state := WindowState{
			Width:      g.windowWidth,
			Height:     g.windowHeight,
			Fullscreen: ebiten.IsFullscreen(),
		}
		// Retrieve window position
		state.X, state.Y = ebiten.WindowPosition()
		saveWindowState(state)
		log.Printf("Window resized: Width=%d, Height=%d, X=%d, Y=%d", g.windowWidth, g.windowHeight, state.X, state.Y)
	}
	return g.windowWidth, g.windowHeight
}

// saveWindowState saves the current window state to a JSON file.
func saveWindowState(state WindowState) {
	file, err := os.Create("window_state.json")
	if err != nil {
		log.Printf("Failed to save window state: %v", err)
		return
	}
	defer file.Close()

	log.Printf("Saving window state: Width=%d, Height=%d, X=%d, Y=%d, Fullscreen=%t",
		state.Width, state.Height, state.X, state.Y, state.Fullscreen)

	if err := json.NewEncoder(file).Encode(state); err != nil {
		log.Printf("Failed to encode window state: %v", err)
		return
	}

	log.Println("Window state saved successfully")
}

// loadWindowState loads the saved window state from a JSON file.
func loadWindowState() WindowState {
	state := WindowState{Width: 1600, Height: 1200, X: 100, Y: 100, Fullscreen: false} // Default values
	file, err := os.Open("window_state.json")
	if err != nil {
		if os.IsNotExist(err) {
			log.Println("window_state.json not found, creating with default values.")
			saveWindowState(state)
			return state
		}
		log.Printf("Failed to load window state: %v", err)
		return state
	}
	defer file.Close()

	if err := json.NewDecoder(file).Decode(&state); err != nil {
		log.Printf("Failed to decode window state: %v", err)
		return state
	}

	// Log the loaded window state for debugging
	log.Printf("Loaded window state: Width=%d, Height=%d, X=%d, Y=%d, Fullscreen=%t",
		state.Width, state.Height, state.X, state.Y, state.Fullscreen)

	return state
}

// main initializes the game and starts the Ebiten game loop.
func main() {
	// Load previous window state
	state := loadWindowState()

	// Set initial window properties
	ebiten.SetWindowSize(state.Width, state.Height)
	ebiten.SetFullscreen(state.Fullscreen)
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	ebiten.SetWindowTitle("Pixel Simulator")
	// Set initial window position
	log.Printf("Setting window position to X=%d, Y=%d", state.X, state.Y)
	ebiten.SetWindowPosition(state.X, state.Y)

	// Create the game
	const cellSize = 5
	game := NewGame(state.Width, state.Height, cellSize)
	game.randomizeGrid(0.1)

	// Handle termination signals to save window state
	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt, syscall.SIGTERM)
		<-c
		os.Exit(0)
	}()

	// Run the game
	if err := ebiten.RunGame(game); err != nil {
		log.Fatalf("Game exited with error: %v", err)
	}
}
