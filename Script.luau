local HttpService = game:GetService("HttpService")

local function serializePose(pose)
	local children = {}
	for _, child in ipairs(pose:GetChildren()) do
		if child:IsA("Pose") then
			table.insert(children, serializePose(child))
		end
	end
	return {
		Name            = pose.Name,
		Weight          = pose.Weight,
		EasingStyle     = pose.EasingStyle.Name,
		EasingDirection = pose.EasingDirection.Name,
		CFrame          = { pose.CFrame:GetComponents() },
		Children        = children,
	}
end

local function deserializePose(poseData, parent)
	local pose = Instance.new("Pose")
	pose.Name           = poseData.Name
	pose.Weight         = poseData.Weight or 1
	pose.EasingStyle    = Enum.PoseEasingStyle[poseData.EasingStyle] or Enum.PoseEasingStyle.Linear
	pose.EasingDirection = Enum.PoseEasingDirection[poseData.EasingDirection] or Enum.PoseEasingDirection.In
	pose.CFrame         = CFrame.new(table.unpack(poseData.CFrame))
	pose.Parent         = parent
	for _, childData in ipairs(poseData.Children or {}) do
		deserializePose(childData, pose)
	end
end

-- Export: KeyframeSequence → JSON string
local function exportAnimation(sequence)
	local data = {}
	for _, keyframe in ipairs(sequence:GetChildren()) do
		if keyframe:IsA("Keyframe") then
			local keyData = { Name = keyframe.Name, Time = keyframe.Time, Poses = {} }
			for _, child in ipairs(keyframe:GetChildren()) do
				if child:IsA("Pose") then
					table.insert(keyData.Poses, serializePose(child))
				end
			end
			table.insert(data, keyData)
		end
	end
	return HttpService:JSONEncode(data)
end

-- Import: JSON string → KeyframeSequence
local function importAnimation(json, name)
	local data = HttpService:JSONDecode(json)
	local sequence = Instance.new("KeyframeSequence")
	sequence.Name = name or "ImportedAnimation"
	for _, keyData in ipairs(data) do
		local keyframe = Instance.new("Keyframe")
		keyframe.Name = keyData.Name
		keyframe.Time = keyData.Time
		for _, poseData in ipairs(keyData.Poses) do
			deserializePose(poseData, keyframe)
		end
		keyframe.Parent = sequence
	end
	return sequence
end

local selection = game:GetService("Selection"):Get()[1]

local json = exportAnimation(selection)
workspace.Export.Source = json

local sequence = importAnimation(workspace.Import.Source, "TEST" .. "_Copy")
sequence.Parent = workspace
game:GetService("Selection"):Set({ sequence })