import "./Button.css"

export default function Button({
  as = "button",
  variant = "primary",
  className = "",
  type = "button",
  children,
  ...props
}) {
  const Tag = as
  const classes = ["app-button", variant, className].filter(Boolean).join(" ")

  if (Tag === "button") {
    return (
      <button type={type} className={classes} {...props}>
        {children}
      </button>
    )
  }

  return (
    <Tag className={classes} {...props}>
      {children}
    </Tag>
  )
}