"""
Generate sample resume PDFs for testing.

Creates 4 varied resume PDFs in tests/fixtures/sample_resumes/:
  1. React Frontend Developer
  2. Python Backend Engineer
  3. Data Scientist
  4. DevOps / Cloud Engineer
"""

from pathlib import Path

import fitz  # PyMuPDF


def _make_pdf(path: Path, text: str) -> None:
    """Create a single-page PDF with the given text."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US Letter
    # Insert text with a simple font
    page.insert_text(
        (50, 72),
        text,
        fontname="helv",
        fontsize=10,
        color=(0, 0, 0),
    )
    doc.save(str(path))
    doc.close()


RESUMES = {
    "Alice_Johnson_React_Developer": """
ALICE JOHNSON
Senior React Frontend Developer
alice.johnson@email.com | San Francisco, CA | +1-555-0101

SUMMARY
Senior Frontend Developer with 6 years of experience building scalable web applications
using React, TypeScript, and Next.js. Proven track record of leading UI teams and
improving application performance by up to 40%.

EXPERIENCE

Senior React Developer — TechCorp Inc. (2022 - Present)
- Led a team of 5 frontend developers building a customer-facing SaaS dashboard
- Architected a micro-frontend system using React and Module Federation
- Reduced bundle size by 35% through code splitting and lazy loading
- Implemented comprehensive unit and integration tests with Jest and React Testing Library
- Mentored 3 junior developers through code reviews and pair programming sessions

React Developer — StartupXYZ (2019 - 2022)
- Built a real-time collaboration tool using React, Socket.io, and WebRTC
- Developed reusable component library with 50+ components using Storybook
- Migrated legacy jQuery codebase to React, improving developer velocity by 60%
- Integrated REST APIs and GraphQL endpoints for data fetching with Apollo Client

Frontend Developer — WebAgency Co. (2018 - 2019)
- Developed responsive marketing websites using React, Gatsby, and Styled Components
- Implemented SEO optimizations resulting in 45% increase in organic traffic
- Collaborated with UX designers to translate Figma mockups into pixel-perfect UIs

TECHNICAL SKILLS
- Languages: JavaScript, TypeScript, HTML5, CSS3, GraphQL
- Frontend: React, Next.js, Redux, Zustand, Tailwind CSS, Material UI, Storybook
- Testing: Jest, React Testing Library, Cypress, Playwright
- Tools: Git, Docker, Webpack, Vite, npm, yarn
- Other: REST APIs, WebSocket, Agile/Scrum, CI/CD (GitHub Actions)

EDUCATION
B.S. Computer Science — University of California, Berkeley (2018)
""",
    "Bob_Smith_Python_Backend": """
BOB SMITH
Python Backend Engineer
bob.smith@email.com | Austin, TX | +1-555-0102

SUMMARY
Backend Engineer with 5 years of experience designing and building robust APIs,
microservices, and data pipelines using Python, Django, and FastAPI. Strong
background in database design and cloud infrastructure.

EXPERIENCE

Senior Backend Engineer — DataFlow Systems (2021 - Present)
- Designed and implemented RESTful APIs serving 10M+ requests/day using FastAPI
- Built an event-driven microservices architecture with RabbitMQ and Docker
- Optimized PostgreSQL query performance, reducing average response time by 70%
- Implemented caching layer with Redis, improving throughput by 3x
- Led database migration from MySQL to PostgreSQL with zero downtime

Backend Developer — CloudSolutions Inc. (2019 - 2021)
- Developed Django REST Framework APIs for an e-commerce platform
- Created automated data pipelines processing 5TB+ of data daily using Apache Airflow
- Implemented authentication and authorization system using JWT and OAuth 2.0
- Wrote comprehensive integration tests achieving 95% code coverage

Junior Python Developer — CodeCraft Labs (2018 - 2019)
- Built internal tools and automation scripts using Python and Bash
- Developed web scraping solutions with Scrapy and BeautifulSoup
- Contributed to open-source Python libraries

TECHNICAL SKILLS
- Languages: Python, SQL, Bash, Go (basic)
- Backend: Django, FastAPI, Flask, Celery, SQLAlchemy, Alembic
- Databases: PostgreSQL, MySQL, Redis, MongoDB, Elasticsearch
- DevOps: Docker, Kubernetes, AWS (EC2, S3, Lambda, RDS), GitHub Actions
- Data: Apache Airflow, pandas, NumPy
- Testing: pytest, coverage, factory_boy, tox

EDUCATION
B.S. Computer Science — University of Texas at Austin (2018)
Certifications: AWS Certified Solutions Architect - Associate
""",
    "Carol_Williams_Data_Scientist": """
CAROL WILLIAMS
Senior Data Scientist
carol.williams@email.com | New York, NY | +1-555-0103

SUMMARY
Data Scientist with 7 years of experience in machine learning, statistical analysis,
and data engineering. Expertise in building production ML models for healthcare
and fintech applications. Published 3 papers in top-tier ML conferences.

EXPERIENCE

Senior Data Scientist — HealthAI Corp (2021 - Present)
- Led development of an NLP system for medical record analysis using BERT and GPT
- Built a patient risk prediction model achieving 92% AUC-ROC score
- Designed A/B testing frameworks for evaluating model performance in production
- Managed a team of 4 data scientists and 2 ML engineers
- Reduced model inference latency by 60% through ONNX optimization and model distillation

Data Scientist — FinTech Analytics (2018 - 2021)
- Developed credit scoring models using XGBoost and LightGBM
- Built real-time fraud detection system processing 1M+ transactions/day
- Created customer segmentation models using K-means and DBSCAN clustering
- Automated feature engineering pipeline reducing model development time by 50%
- Presented ML insights to C-suite stakeholders and board members

ML Research Intern — Stanford AI Lab (2017 - 2018)
- Conducted research on transformer architectures for time-series forecasting
- Implemented novel attention mechanisms improving forecast accuracy by 15%
- Co-authored paper accepted at NeurIPS 2018

TECHNICAL SKILLS
- Languages: Python, R, SQL, Julia
- ML/DL: scikit-learn, TensorFlow, PyTorch, Hugging Face, XGBoost, LightGBM
- NLP: spaCy, NLTK, transformers, LangChain
- Data: pandas, NumPy, SciPy, Polars, Apache Spark
- Visualization: Matplotlib, Seaborn, Plotly, Tableau
- MLOps: MLflow, Docker, SageMaker, Kubeflow, DVC
- Tools: Git, Jupyter, Airflow, dbt

EDUCATION
Ph.D. Statistics — Stanford University (2018)
M.S. Data Science — Columbia University (2016)
B.S. Mathematics — MIT (2014)
""",
    "David_Lee_DevOps_Engineer": """
DAVID LEE
DevOps / Cloud Engineer
david.lee@email.com | Seattle, WA | +1-555-0104

SUMMARY
DevOps Engineer with 4 years of experience in cloud infrastructure, CI/CD pipelines,
containerization, and infrastructure as code. Passionate about automating everything
and building reliable, scalable systems on AWS and GCP.

EXPERIENCE

DevOps Engineer — CloudScale Inc. (2022 - Present)
- Managed Kubernetes clusters serving 200+ microservices across 3 environments
- Built CI/CD pipelines using GitHub Actions, ArgoCD, and Helm charts
- Reduced infrastructure costs by 40% through right-sizing and spot instance strategies
- Implemented infrastructure as code using Terraform managing 500+ AWS resources
- Designed monitoring and alerting systems using Prometheus, Grafana, and PagerDuty
- Achieved 99.95% uptime SLA for production services

Cloud Engineer — StartupOps (2020 - 2022)
- Migrated on-premises infrastructure to AWS, reducing operational overhead by 60%
- Containerized 30+ applications using Docker and deployed to ECS Fargate
- Set up automated backups and disaster recovery procedures
- Implemented secrets management using AWS Secrets Manager and HashiCorp Vault
- Built self-service infrastructure provisioning platform using Terraform and Go

Systems Administrator — TechHosting Co. (2019 - 2020)
- Managed 100+ Linux servers (Ubuntu, CentOS) in production environment
- Automated server provisioning with Ansible playbooks
- Configured Nginx, HAProxy load balancers, and SSL/TLS certificates
- On-call rotation for 24/7 production support

TECHNICAL SKILLS
- Cloud: AWS (EC2, S3, Lambda, EKS, RDS, CloudFront, IAM), GCP (GKE, BigQuery)
- Containers: Docker, Kubernetes, Helm, ArgoCD, ECS
- IaC: Terraform, Ansible, CloudFormation, Pulumi
- CI/CD: GitHub Actions, Jenkins, GitLab CI, ArgoCD
- Monitoring: Prometheus, Grafana, Datadog, ELK Stack, PagerDuty
- Languages: Python, Bash, Go (intermediate)
- Networking: VPC, Route 53, CloudFront, VPN, TLS/SSL
- Databases: RDS PostgreSQL, DynamoDB, ElastiCache, MongoDB Atlas

EDUCATION
B.S. Information Technology — University of Washington (2019)
Certifications: AWS Certified DevOps Engineer - Professional, CKA (Kubernetes)
""",
}


def main() -> None:
    output_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_resumes"
    output_dir.mkdir(parents=True, exist_ok=True)

    for stem, text in RESUMES.items():
        pdf_path = output_dir / f"{stem}.pdf"
        _make_pdf(pdf_path, text.strip())
        print(f"  Created: {pdf_path.name}")

    print(f"\nGenerated {len(RESUMES)} sample resume PDFs in {output_dir}")


if __name__ == "__main__":
    main()