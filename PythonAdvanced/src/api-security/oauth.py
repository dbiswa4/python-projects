import base64
import hashlib
import hmac
import json
import time
import urlparse
import requests
import sys


def hmac_sha256(key, msg, encode_output=False):

    message = bytes(msg).encode('utf-8')
    secret = bytes(key).encode('utf-8')

    signature = hmac.new(secret, message, digestmod=hashlib.sha256).digest()

    return base64.b64encode(signature) if encode_output else signature

def get_access_token(client_config):
    """
    Returns an access token for the given client credentials

    :param client_config: A dict with the environment variables. Is different on QA/PROD
    :return: the oauth access token for the client
    """

    client_id = client_config['CLIENT_ID']
    print 'client_id : ', client_id

    client_secret = client_config['CLIENT_SECRET']
    realm = client_config['REALM']
    base_url = client_config['BASE_URL']
    scope = client_config['SCOPE']
    access_token_url_path = 'identity/oauth2/access_token'

    jwt_header = json.dumps({
        "typ": "JWT",
        "alg": "HS256",
    })

    issue_time = int(time.time())  # Seconds since epoch
    expiry_time = issue_time + 600
    aud = urlparse.urljoin(base_url, '{path}?realm={realm}'.format(path=access_token_url_path, realm=realm))

    print 'aud : ', aud

    jwt_body = {
        "iss": client_id,
        "sub": client_id,
        "aud": aud,
        "exp": expiry_time,
        "iat": issue_time,
    }

    jwt_body = json.dumps(jwt_body)

    jwt_signing_string = base64.b64encode(jwt_header) + '.' + base64.b64encode(jwt_body)

    signature = hmac_sha256(client_secret, jwt_signing_string)

    jwt_signature = base64.b64encode(signature)

    client_assertion = jwt_signing_string + '.' + jwt_signature

    data = {
        'grant_type': 'client_credentials',
        'scope': scope,
        'client_assertion_type': 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer',
        'client_assertion': client_assertion,
        'realm': realm,
    }
    print 'data : ', data
    print "getting access token"
    resp = requests.post(urlparse.urljoin(base_url, access_token_url_path), data=data)

    print 'resp : ', resp

    result = resp.json()
    print 'result : ', result

    return result['access_token']

if __name__ == '__main__':
    print 'Get oauth token'

    if len(sys.argv) < 3:
        print 'Not enough arguments'
        exit(1)

    config = {}
    config['CLIENT_ID'] = sys.argv[1]
    config['CLIENT_SECRET'] = sys.argv[2]
    config['REALM'] = 'myrealm'
    config['BASE_URL'] = 'https://id-uat2.corp.techknowera.com'
    #config['BASE_URL'] = 'https://id.techknowera.aol.com'
    config['SCOPE'] = 'techknowera-api'

    print 'config : \n', config
    print config['CLIENT_ID']
    get_access_token(config)