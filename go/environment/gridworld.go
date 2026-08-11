package environment

const (
	ActionUp = iota
	ActionDown
	ActionLeft
	ActionRight
)

type Position struct {
	Row int
	Col int
}

type Config struct {
	Rows           int
	Cols           int
	StepReward     float64
	ObstacleReward float64
	GoalReward     float64
}

type GridWorld struct {
	Config    Config
	Start     Position
	Goal      Position
	Agent     Position
	Obstacles map[Position]bool
	Done      bool
}

func NewGridWorld(config Config) *GridWorld {
	env := &GridWorld{
		Config: config,
		Start: Position{
			Row: 0,
			Col: 0,
		},
		Goal: Position{
			Row: config.Rows - 1,
			Col: config.Cols - 1,
		},
		Obstacles: make(map[Position]bool),
		Done:      false,
	}

	env.Obstacles[Position{1, 1}] = true
	env.Obstacles[Position{1, 3}] = true
	env.Obstacles[Position{2, 3}] = true
	env.Obstacles[Position{3, 1}] = true
	env.Obstacles[Position{4, 1}] = true
	env.Obstacles[Position{4, 4}] = true

	env.Reset()

	return env
}

func (env *GridWorld) Reset() Position {
	env.Agent = env.Start
	env.Done = false

	return env.Agent
}

func (env *GridWorld) Step(action int) (Position, float64, bool) {
	if env.Done {
		panic("episode has already finished")
	}

	rowDelta, colDelta := actionDelta(action)

	newPosition := Position{
		Row: env.Agent.Row + rowDelta,
		Col: env.Agent.Col + colDelta,
	}

	if !env.IsValidPosition(newPosition) {
		return env.Agent, env.Config.ObstacleReward, false
	}

	env.Agent = newPosition

	if env.Agent == env.Goal {
		env.Done = true
		return env.Agent, env.Config.GoalReward, true
	}

	return env.Agent, env.Config.StepReward, false
}

func (env *GridWorld) IsValidPosition(position Position) bool {
	if position.Row < 0 || position.Row >= env.Config.Rows {
		return false
	}

	if position.Col < 0 || position.Col >= env.Config.Cols {
		return false
	}

	if env.Obstacles[position] {
		return false
	}

	return true
}

func actionDelta(action int) (int, int) {
	switch action {
	case ActionUp:
		return -1, 0
	case ActionDown:
		return 1, 0
	case ActionLeft:
		return 0, -1
	case ActionRight:
		return 0, 1
	default:
		panic("invalid action")
	}
}
