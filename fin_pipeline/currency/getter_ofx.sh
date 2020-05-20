##
# example:
# bash ./getter_ofx.sh USD CNY

curl -H "Accept: application/json, text/plain, */*" -H "Accept-Encoding: gzip, deflate, br" -H "Accept-Language: en-US" -H "Cache-Control: no-cache" -H "Connection: keep-alive" -H "Host: api.ofx.com" -H "Origin: https://www.ofx.com" -H "Pragma: no-cache" -H "Referer: https://www.ofx.com/en-us/forex-news/historical-exchange-rates/" -H "Sec-Fetch-Dest: empty" -H "Sec-Fetch-Mode: cors" -H "Sec-Fetch-Site: same-site" -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36" -v "https://api.ofx.com/PublicSite.ApiService/SpotRateHistory/allTime/$1/$2?DecimalPlaces=6&ReportingInterval=daily" > ./currency_$1_to_$2.json