FROM continuumio/anaconda3:4.4.0
MAINTAINER UNP, https://unp.education
COPY ./future_sale /usr/local/python
EXPOSE 5001
WORKDIR /usr/local/python
RUN pip install -r requirements.txt
CMD python app.py
