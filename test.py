import requests
import json

uri = "https://voyfinance.instance.kyc-chain.com/integrations/v3/scope/voyfinance/data-checks/entity?countryCodes=GB&countryCodes=US&name=Cargill%20Incorporated"
# uri = "https://voyfinance.instance.kyc-chain.com/integrations/v3/scope/voyfinance/wallet-screening/"
# uri = "https://voyfinance.instance.kyc-chain.com/integrations/v3/scope/voyfinance/risk-scoring/scorecard/"
token = "KEY"

# HTTP header fields to set
headers = {
    "apiKey": token,
    "accept": "application/json",
}
# make request
r = requests.get(uri, headers=headers, verify=False)

# print the result
# print(r.keys())
# for key in r.text:
#     print(key)

res = r.json()
# res = res["complyAdvantageEntities"]

print(json.dumps(res, indent=4, sort_keys=True))
# print(res.keys())


file1 = open("data_checks.txt", "w")
file1.writelines(str(res))
file1.close()
