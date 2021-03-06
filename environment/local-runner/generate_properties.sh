#!/bin/bash
cat <<- EOT > generated.properties
	render-to-screen=false
	render-to-screen-sync=
	render-to-screen-size=
	render-to-screen-tick=

	results-file=result.txt
	log-file=game.log

	team-size=1
	player-count=4

	p1-type=Local
	p2-type=Empty
	p3-type=Empty
	p4-type=Empty

	p1-name=
	p2-name=
	p3-name=
	p4-name=

	swap-car-types=false
	disable-car-collision=true
	# default, _fdoke, _tyamgin, _ud1, map01 - map21.
	map=map0$(( (RANDOM % 9) + 1 ))

	base-adapter-port=31001
	seed=$(( (RANDOM % 100) + 1 ))
	plugins-directory=
EOT
