SELECT DISTINCT t1.vplayerid AS gplayerid
FROM dws_argothek_ce1_cbt2_vplayerid_suserid_di t1
LEFT JOIN dim_extract_311381_conf t2 ON t1.vplayerid = t2.vplayerid
WHERE t1.dtstatdate BETWEEN '20250717' AND '20250723'
AND t1.itemp2 > 0
AND t2.vplayerid IS NULL
ORDER BY t1.vplayerid