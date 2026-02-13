FROM debian:latest

RUN apt-get update
RUN apt-get install -y libstdc++6

COPY prompt /app/prompt
COPY context.txt /app/context.txt
COPY models/ /app/models/

ENTRYPOINT ["bash"]
