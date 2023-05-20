FROM python:3

RUN apt-get update \
    && apt-get install -y xinetd gnupg

RUN touch /var/log/xinetdlog

ENV USER diabetes

WORKDIR /home/$USER

RUN useradd $USER

COPY diabetes.csv /home/$USER/
COPY diabetes_model.py /home/$USER/
COPY requirements.txt /home/$USER/
COPY run.sh /home/$USER/

RUN pip install -r requirements.txt

COPY $USER.xinetd /etc/xinetd.d/$USER

RUN chown -R root:$USER /home/$USER
RUN chmod -R 550 /home/$USER

EXPOSE 1588

CMD service xinetd start && sleep 2 && tail -f /var/log/xinetdlog
