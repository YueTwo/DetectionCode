nohup /root/autodl-tmp/frp_0.56.0_linux_amd64/frpc -c /root/autodl-tmp/frp_0.56.0_linux_amd64/frpc.toml 2>1 &
nohup snowflake_start_server --address=127.0.0.1 --port=8910 --dc=2 --worker=1 --log_file_prefix=/tmp/pysnowflask.log>/dev/null &
python /root/flask/RequestGet.py
