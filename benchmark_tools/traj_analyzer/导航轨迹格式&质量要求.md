## 前提
* 场景符合要求(几何-视觉-语义)
* 任务符合要求(随记起点-随记类别)
* 最短路径计算是正确的

## 开源HD轨迹
[pirlnav](https://github.com/Ram81/pirlnav)

## 轨迹规范和开源HD轨迹关系
* 格式上和开源HD拉齐
* 质量上向HD看齐，HD存在的问题一并解决

# 轨迹问题汇总
备注（√表示是问题，×表示不是问题，？表示不确定，高中低表示优先级）

* [√高]起始action不是STOP
* [√高]终点action不是STOP
* [?]轨迹长度超出500

* [√高]探索路径复杂
* [√]同一轨迹下存在连续低头的视角
* [√]视野内存在目标类别，但未走过去
* [√]经过viewpoint，但是没有成功找到
* [×]整段轨迹存在上下楼
    任务不涉及爬楼
* [×]轨迹中存在LOOK_UP，LOOK_DOWN
    动作空间没有LOOK_UP, LOOK_DOWN
* [?]停位时视野内无目标 or 占比较低
* [?]频繁LOOK_UP, LOOK_DOWN, TURN_LEFT, TURN_RIGHT
* [?]在中途停止不动大概3秒，过好一会再动起来
* [?]起点附近有目标类别，但是没有360环视，导致错过
    * 探索策略的选择相关
* [?]同楼层存在目标类别，但是找到了其他楼层的
    * 探索策略的选择相关
* [?]路过门口没有进行探索
    * 探索策略的选择相关


## 轨迹格式要求
```
{
    "episodes": [
        {
            "episode_id": "xxx",
            "scene_id": "xxx",
            "start_position": [xxx,xxx,xxx],
            "start_rotation": [xxx,xxx,xxx,xxx],
            "object_category": "cushion",
            "info": {
                "geodesic_distance": 9.33001,
                "euclidean_distance": 4.61717,
                "closest_goal_object_id": 24,
            },
            "reference_replay": [
                {
                    "action": "STOP",
                    "agent_state": {
                        "position": [xxx,xxx,xxx],
                        "rotation": [xxx,xxx,xxx,xxx]
                    }
                },
                {
                    "action": "[TURN_LEFT | TURN_RIGHT | MOVE_FORWARD | LOOK_UP | LOOK_DOWN]",
                    "agent_state": {
                        "position": [xxx,xxx,xxx],
                        "rotation": [xxx,xxx,xxx,xxx]
                    }
                },
                {
                    "action": "STOP",
                    "agent_state": {
                        "position": [xxx,xxx,xxx],
                        "rotation": [xxx,xxx,xxx,xxx]
                    }
                }
            ]
        }
    ]
}
```
* [√高]起始waypoint的action是STOP
    * HD所有场景数据已确认
* [√高]终点waypoint的action是STOP
    * HD所有场景数据已确认
* [√]中途waypoint的action的取值范围是[TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, LOOK_UP, LOOK_DOWN]
    * HD所有场景数据已确认
* [?]waypoint的长度小于500，仅针对自采轨迹
    * HD-77K轨迹中，轨迹长度在[500,1000]的2352条; >1000的319条
    * L3最长的是500

## 轨迹质量要求
* [√高]spl的值应位于[min,max]
    ![spl-hd](image.png)
    ![spl-hd-l3](image-3.png)
    ![spl-v2-l3](image-2.png)
* [√]向上向下看的占比应小于xxx
* [√]中途目标类别任一实例iou应小于xxx

* [?]整段轨迹在同一个楼层
* [?]去除LOOK_UP和LOOK_DOWN
* [?]终点视角目标类别占比大于xxx
* [?]LOOK_* TURN_*的占比

##
HD格式分析，L3MVN格式上与其拉齐
    L3起始action调整为STOP
    L3有部分step长度为500，最后一个action不是STOP
        存量轨迹做筛选
        后续的轨迹，success & action = STOP

模型评测&轨迹分析
    done HD&自采轨迹的SPL分析
    todo 轨迹分析更细节——轨迹分析
        spl: 类别；导航区间；导航复杂度

        向上看向下看区间的占比

        中途目标类别任一实例iou应小于xxx