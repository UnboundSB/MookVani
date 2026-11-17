import "./PageHeader.css"

export default function PageHeader({ title, description }) {
  return (
    <header className="page-header">
      <h1>{title}</h1>
      {description ? <p>{description}</p> : null}
    </header>
  )
}