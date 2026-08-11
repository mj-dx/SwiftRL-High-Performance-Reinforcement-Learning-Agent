package main

import (
	"fmt"

	"github.com/mj-dx/SwiftRL/go/environment"
)

func main() {
	config := environment.Config{
		Rows:           6,
		Cols:           6,
		StepReward:     -1.0,
		ObstacleReward: -10.0,
		GoalReward:     100.0,
	}

	env := environment.NewGridWorld(config)

	state := env.Reset()

	fmt.Printf("Initial state: %+v\n", state)

	actions := []int{
		environment.ActionRight,
		environment.ActionRight,
		environment.ActionDown,
		environment.ActionDown,
	}

	for _, action := range actions {
		state, reward, done := env.Step(action)

		fmt.Printf(
			"Action: %d | State: %+v | Reward: %.1f | Done: %t\n",
			action,
			state,
			reward,
			done,
		)

		if done {
			break
		}
	}
}
