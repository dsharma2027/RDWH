-- Extract File name from the File Destination.

Declare @addr varchar (1000)
Set @addr = 'C:\SQL2019\Express_ENU\PackageId.dat'


SELECT LEFT(@addr,LEN(@addr) - charindex('\',reverse(@addr),1) + 1) AS p_name, 
       RIGHT(@addr, CHARINDEX('\', REVERSE(@addr))-1)  AS f_name

SELECT LEFT(@addr,LEN(@addr) - charindex('\',reverse(@addr)) + 1) AS p_name, 
       RIGHT(@addr, CHARINDEX('\', REVERSE(@addr))-1)  AS f_name


