-- DWS sql 
-- ******************************************************************** --
-- author: CTO_TI_FBSYJG
-- create time: 2025/10/28 15:09:40 GMT+08:00
-- ******************************************************************** --
DROP FOREIGN TABLE IF EXISTS ${job_id};

SET
	search_path = public;

CREATE FOREIGN TABLE ${job_id} (
	scene_name CHARACTER VARYING(100),
	uuid_array TEXT
) SERVER gsmpp_server OPTIONS (
	access_key 'IP5YP0WXOJG4MAUBZEEP',
	DELIMITER '|',
	ENCODING 'utf-8',
	encrypt 'on',
	FORMAT 'text',
	LOCATION ${location},
	secret_access_key 'encryptstrHkKgRXEIpv/wZsv6DPif0O+2CcYixeF/1vxoVjtw8DXyJfHCyf2H6Q90PAWgaRdXwMrPf66rU+X8+ucV1rLHUWl2gjWWxHg0INojEf992gys5w=='
) WRITE ONLY;
