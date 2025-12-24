-- DWS sql 
-- ******************************************************************** --
-- author: CTO_TI_FBSYJG
-- create time: 2025/09/23 15:25:08 GMT+08:00
-- ******************************************************************** --
CREATE TEMPORARY TABLE
	scene_ids AS (
		SELECT
			task_datasets_objectnav.scene_name,
			COUNT(traj_datasets_objectnav.id) AS id_count,
			STRING_AGG(traj_datasets_objectnav.id, ',') AS id_list
		FROM
			public.task_datasets_objectnav
			JOIN public.traj_datasets_objectnav ON traj_datasets_objectnav.task_id = task_datasets_objectnav.id
		WHERE
			traj_datasets_objectnav.gen_traj_method = '${gen_traj_method}'
			AND traj_datasets_objectnav.success = ${success}
			AND traj_datasets_objectnav.spl BETWEEN ${spl_min} AND ${spl_max}
			AND traj_datasets_objectnav.experiment_name = '${experiment_name}'
			AND traj_datasets_objectnav.recipe_tags @> ARRAY['${traj_recipe_tag}']
			AND task_datasets_objectnav.scene_type = '${scene_type}'
			AND task_datasets_objectnav.split = '${split}'
			AND (
				'${scene_name}' = 'ALL_SCENE'
				OR task_datasets_objectnav.scene_name = '${scene_name}'
			)
			AND (
				'${object_category}' = 'ALL_CATEGORY'
				OR task_datasets_objectnav.object_category = '${object_category}'
			)
			AND task_datasets_objectnav.geodesic_distance BETWEEN ${geodesic_distance_min} AND ${geodesic_distance_max}
			AND task_datasets_objectnav.nav_complexity_ratio BETWEEN ${nav_complexity_ratio_min} AND ${nav_complexity_ratio_max}
			AND task_datasets_objectnav.recipe_tags @> ARRAY['${task_recipe_tag}']
		GROUP BY
			task_datasets_objectnav.scene_name
	);

INSERT INTO
	${job_id}
SELECT
	scene_name,
	id_list
FROM
	scene_ids;

SELECT
	scene_name,
	id_count
FROM
	scene_ids;