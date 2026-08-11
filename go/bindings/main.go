package main

/*
#include <stdint.h>
*/
import "C"

import (
	"unsafe"

	"github.com/mj-dx/SwiftRL/go/environment"
)

var env *environment.GridWorld

//export CreateEnvironment
func CreateEnvironment(
	rows C.int,
	cols C.int,
	stepReward C.double,
	obstacleReward C.double,
	goalReward C.double,
) {
	config := environment.Config{
		Rows:           int(rows),
		Cols:           int(cols),
		StepReward:     float64(stepReward),
		ObstacleReward: float64(obstacleReward),
		GoalReward:     float64(goalReward),
	}

	env = environment.NewGridWorld(config)
}

//export ResetEnvironment
func ResetEnvironment() C.int {
	if env == nil {
		return -1
	}

	state := env.Reset()

	return C.int(encodePosition(state))
}

//export StepEnvironment
func StepEnvironment(
	action C.int,
	reward *C.double,
	done *C.int,
) C.int {
	if env == nil {
		return -1
	}

	state, stepReward, isDone := env.Step(int(action))

	*reward = C.double(stepReward)

	if isDone {
		*done = 1
	} else {
		*done = 0
	}

	return C.int(encodePosition(state))
}

//export RunBatch
func RunBatch(
	actions *C.int,
	actionCount C.int,
	states *C.int,
	rewards *C.double,
	dones *C.int,
) C.int {
	if env == nil {
		return -1
	}

	count := int(actionCount)

	actionSlice := unsafe.Slice(actions, count)
	stateSlice := unsafe.Slice(states, count)
	rewardSlice := unsafe.Slice(rewards, count)
	doneSlice := unsafe.Slice(dones, count)

	for i := 0; i < count; i++ {
		state, reward, done := env.Step(
			int(actionSlice[i]),
		)

		stateSlice[i] = C.int(
			encodePosition(state),
		)

		rewardSlice[i] = C.double(reward)

		if done {
			doneSlice[i] = 1
		} else {
			doneSlice[i] = 0
		}

		if done && i+1 < count {
			env.Reset()
		}
	}

	return C.int(count)
}

func encodePosition(position environment.Position) int {
	return position.Row*env.Config.Cols + position.Col
}

func main() {
	_ = unsafe.Pointer(nil)
}
