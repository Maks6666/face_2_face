-- table with indexes of detected objects

-- CREATE TABLE object_detection_idx (
--     basic_id SERIAL PRIMARY KEY,
-- 	   object_id_idx INTEGER NOT NULL,
--     tracked_object_idx INTEGER NOT NULL,
--     age_category_idx INTEGER NOT NULL,
--     gender_idx INTEGER NOT NULL,
--     race_idx INTEGER NOT NULL,
--     face_idx INTEGER NOT NULL,
--     status VARCHAR(100) NOT NULL,
-- 	   total_obj INTEGER NOT NULL,
--     time_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
--
-- );



-- table with visually-understandable information about detected objects

-- CREATE TABLE object_detection (
--     object_id SERIAL PRIMARY KEY,
--     tracked_object_id INTEGER NOT NULL,
--     object_class VARCHAR NOT NULL,
--     age_category VARCHAR NOT NULL,
--     gender VARCHAR NOT NULL,
--     race VARCHAR NOT NULL,
--     face VARCHAR NOT NULL,
--     last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
-- );



-- DROP TABLE object_detection
-- DROP TABLE object_detection_idx

-- SELECT * FROM object_detection
-- SELECT * FROM object_detection_idx


-- DELETE FROM object_detection;
-- DELETE FROM object_detection_idx