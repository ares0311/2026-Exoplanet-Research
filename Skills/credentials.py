import os


def get_atlas_token() -> str:
    token = os.environ.get("ATLAS_TOKEN")
    if not token:
        raise OSError("ATLAS_TOKEN environment variable not set")
    return token


def get_ztf_credentials() -> tuple[str, str]:
    username = os.environ.get("ZTF_IRSA_USERNAME")
    password = os.environ.get("ZTF_IRSA_PASSWORD")
    if not username or not password:
        raise OSError(
            "ZTF_IRSA_USERNAME and ZTF_IRSA_PASSWORD must both be set"
        )
    return username, password
