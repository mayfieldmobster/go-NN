package core

import "NN/activation_funcs"
import "math/rand"
import "NN/loss_functions"
import "NN/core"

type model struct {
	layer1 []core.Layer
	layer2 []core.Layer
	layer3 []core.Layer
	layer4 []core.Layer
	loss_function string
}

