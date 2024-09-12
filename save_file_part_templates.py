header = """
ship = {save_name}
version = 1.12.5
description = {description}
type = {building}
size = {width},{height},{length}
steamPublishedFileId = 0
persistentId = {save_persist_id}
rot = 0,0,0,0
missionFlag = Squad/Flags/minimalistic
vesselType = Debris
OverrideDefault = False,False,False,False
OverrideActionControl = 0,0,0,0
OverrideAxisControl = 0,0,0,0
OverrideGroupNames = ,,,
"""

part = """\
PART
{{
	part = {part_name}_{part_id}
	partName = Part
	persistentId = {persist_id}
	pos = {x},{y},{z}
	attPos = 0,0,0
	attPos0 = {attPos0_0},{attPos0_1},{attPos0_2}
	rot = 0,0,0,1
	attRot = 0,0,0,1
	attRot0 = 0,0,0,1
	mir = 1,1,1
	symMethod = Radial
	autostrutMode = Off
	rigidAttachment = False
	istg = {stage}
	resPri = 0
	dstg = {decoupler_stage}
	sidx = {stage_index}
	sqor = {sqor}
	sepI = {sepI}
	attm = {attach_mode}
	sameVesselCollision = False
	modCost = 0
	modMass = 0
	modSize = 0,0,0
	{link}
	attN = top,{attached_top}_0|{node_top}|0_0|1|0_0|{node_top}|0_0|1|0
	attN = bottom,{attached_bottom}_0|{node_bottom}|0_0|-1|0_0|{node_bottom}|0_0|-1|0
	EVENTS
	{{
	}}
	ACTIONS
	{{
		ToggleSameVesselInteraction
		{{
			actionGroup = None
			wasActiveBeforePartWasAdjusted = False
		}}
		SetSameVesselInteraction
		{{
			actionGroup = None
			wasActiveBeforePartWasAdjusted = False
		}}
		RemoveSameVesselInteraction
		{{
			actionGroup = None
			wasActiveBeforePartWasAdjusted = False
		}}
	}}
	PARTDATA
	{{
	}}
    {blocks}
}}
"""

liquid_fuel = """
    RESOURCE
	{{
		name = LiquidFuel
		amount = {amt}
		maxAmount = {max}
		flowState = True
		isTweakable = True
		hideFlow = False
		isVisible = True
		flowMode = Both
	}}
"""
oxidizer = """
	RESOURCE
	{{
		name = Oxidizer
		amount = {amt}
		maxAmount = {max}
		flowState = True
		isTweakable = True
		hideFlow = False
		isVisible = True
		flowMode = Both
	}}
"""
